use glow::*;
use std::sync::Arc;
use image::{RgbaImage};

pub struct Renderer {
    gl: Arc<Context>,
    program: Program,
    vao: VertexArray,
    _vbo: Buffer,
    texture_a: Texture,
    texture_b: Texture,
    ascii_texture: Texture,
    pub ai_bloom: f32,
    pub ai_warp: f32,
    pub ai_kaleido: f32,
}

impl Renderer {
    pub fn new(gl: Arc<Context>, width: u32, height: u32) -> Result<Self, anyhow::Error> {
        unsafe {
            let program = create_program(&gl)?;
            let vao = gl.create_vertex_array().map_err(anyhow::Error::msg)?;
            gl.bind_vertex_array(Some(vao));

            let vbo = gl.create_buffer().map_err(anyhow::Error::msg)?;
            gl.bind_buffer(ARRAY_BUFFER, Some(vbo));
            let vertices: [f32; 16] = [
                -1.0, -1.0, 0.0, 0.0,
                1.0, -1.0, 1.0, 0.0,
                -1.0,  1.0, 0.0, 1.0,
                1.0,  1.0, 1.0, 1.0,
            ];
            gl.buffer_data_u8_slice(ARRAY_BUFFER, bytemuck::cast_slice(&vertices), STATIC_DRAW);

            let vert_attr = gl.get_attrib_location(program, "in_vert").unwrap();
            gl.enable_vertex_attrib_array(vert_attr);
            gl.vertex_attrib_pointer_f32(vert_attr, 2, FLOAT, false, 16, 0);

            let uv_attr = gl.get_attrib_location(program, "in_uv").unwrap();
            gl.enable_vertex_attrib_array(uv_attr);
            gl.vertex_attrib_pointer_f32(uv_attr, 2, FLOAT, false, 16, 8);

            let texture_a = create_texture(&gl, width, height)?;
            let texture_b = create_texture(&gl, width, height)?;
            let ascii_texture = create_ascii_atlas(&gl)?;
            
            gl.use_program(Some(program));
            gl.uniform_2_f32(gl.get_uniform_location(program, "u_px").as_ref(), 1.0 / width as f32, 1.0 / height as f32);
            gl.uniform_2_f32(gl.get_uniform_location(program, "u_resolution").as_ref(), width as f32, height as f32);

            Ok(Self {
                gl,
                program,
                vao,
                _vbo: vbo,
                texture_a,
                texture_b,
                ascii_texture,
                ai_bloom: 0.0,
                ai_warp: 0.0,
                ai_kaleido: 0.0,
            })
        }
    }

    pub unsafe fn render(&mut self, audio: &crate::audio::AudioFeatures, crossfade: f32, _time: f64, _palette: &[String]) {
        let gl = &self.gl;
        gl.use_program(Some(self.program));

        // Optimized texture binding - only 2 textures instead of 3
        gl.active_texture(TEXTURE0);
        gl.bind_texture(TEXTURE_2D, Some(self.texture_a));
        gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_texture_a").as_ref(), 0);

        gl.active_texture(TEXTURE1);
        gl.bind_texture(TEXTURE_2D, Some(self.texture_b));
        gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_texture_b").as_ref(), 1);

        gl.active_texture(TEXTURE2);
        gl.bind_texture(TEXTURE_2D, Some(self.ascii_texture));
        gl.uniform_1_i32(gl.get_uniform_location(self.program, "u_ascii_atlas").as_ref(), 2);

        // Reduced audio uniforms - only essential ones
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_crossfade").as_ref(), crossfade);
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_rms").as_ref(), audio.smoothed_rms / 3000.0);
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_bass").as_ref(), audio.bass / 2000.0);
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_high").as_ref(), audio.high / 2000.0);
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_centroid").as_ref(), audio.centroid);
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_beat").as_ref(), if audio.beat { 1.0 } else { 0.0 });

        // Essential uniforms for visibility and animation
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_time").as_ref(), _time as f32);
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_audio_presence").as_ref(), 1.0); // Always show visuals
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_ascii_blend").as_ref(), 1.0);    // Enable ASCII (Full blend)
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_cell_size").as_ref(), 10.0);    // Set cell size (Denser for precision)
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_splat_blend").as_ref(), 0.0);    // Disable Gaussian Splatting

        // AI-controlled effects
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_ai_bloom").as_ref(), self.ai_bloom);
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_ai_warp").as_ref(), self.ai_warp);
        gl.uniform_1_f32(gl.get_uniform_location(self.program, "u_ai_kaleido").as_ref(), self.ai_kaleido);

        gl.clear(COLOR_BUFFER_BIT);
        gl.bind_vertex_array(Some(self.vao));
        gl.draw_arrays(TRIANGLE_STRIP, 0, 4);
    }

    pub unsafe fn update_texture(&mut self, is_a: bool, image: &RgbaImage) {
        let tex = if is_a { self.texture_a } else { self.texture_b };
        self.gl.bind_texture(TEXTURE_2D, Some(tex));
        
        // Optimized texture upload - use sub-image update if possible
        self.gl.tex_image_2d(
            TEXTURE_2D, 
            0, 
            RGBA as i32, 
            image.width() as i32, 
            image.height() as i32, 
            0, 
            RGBA, 
            UNSIGNED_BYTE, 
            Some(image)
        );
        
        // Set texture parameters for optimal performance
        self.gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MIN_FILTER, LINEAR as i32);
        self.gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MAG_FILTER, LINEAR as i32);
        self.gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_S, CLAMP_TO_EDGE as i32);
        self.gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_WRAP_T, CLAMP_TO_EDGE as i32);
    }
}

unsafe fn create_program(gl: &Context) -> Result<Program, anyhow::Error> {
    let program = gl.create_program().map_err(anyhow::Error::msg)?;

    let vs = gl.create_shader(VERTEX_SHADER).map_err(anyhow::Error::msg)?;
    gl.shader_source(vs, include_str!("shaders/vj.vert"));
    gl.compile_shader(vs);
    if !gl.get_shader_compile_status(vs) {
        return Err(anyhow::anyhow!("VS: {}", gl.get_shader_info_log(vs)));
    }

    let fs = gl.create_shader(FRAGMENT_SHADER).map_err(anyhow::Error::msg)?;
    // Use standard fragment shader (optimized one is incomplete)
    gl.shader_source(fs, include_str!("shaders/vj.frag"));
    gl.compile_shader(fs);
    if !gl.get_shader_compile_status(fs) {
        return Err(anyhow::anyhow!("FS: {}", gl.get_shader_info_log(fs)));
    }

    gl.attach_shader(program, vs);
    gl.attach_shader(program, fs);
    gl.link_program(program);
    if !gl.get_program_link_status(program) {
        return Err(anyhow::anyhow!("Link: {}", gl.get_program_info_log(program)));
    }

    Ok(program)
}

unsafe fn create_texture(gl: &Context, width: u32, height: u32) -> Result<Texture, anyhow::Error> {
    let tex = gl.create_texture().map_err(anyhow::Error::msg)?;
    gl.bind_texture(TEXTURE_2D, Some(tex));
    gl.tex_image_2d(TEXTURE_2D, 0, RGBA as i32, width as i32, height as i32, 0, RGBA, UNSIGNED_BYTE, None);
    gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MIN_FILTER, LINEAR as i32);
    gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MAG_FILTER, LINEAR as i32);
    Ok(tex)
}

unsafe fn create_ascii_atlas(gl: &Context) -> Result<Texture, anyhow::Error> {
    let tex = gl.create_texture().map_err(anyhow::Error::msg)?;
    gl.bind_texture(TEXTURE_2D, Some(tex));
    
    // Create structured ASCII-like patterns 256x16 (16 chars of 16x16)
    let mut data = Vec::with_capacity(256 * 16 * 4);
    // Initialize with 0
    for _ in 0..(256 * 16 * 4) { data.push(0); }

    let width = 256;
    let char_w = 16;
    let char_h = 16;

    // Helper to set pixel alpha in RGBA buffer
    let mut set_pixel = |x: usize, y: usize, val: u8| {
        let idx = (y * width + x) * 4;
        if idx < data.len() {
            data[idx] = val;     // R
            data[idx+1] = val;   // G
            data[idx+2] = val;   // B
            data[idx+3] = val;   // A
        }
    };

    for i in 0..16 { // 16 characters
        let off_x = i * char_w;
        for y in 0..char_h {
            for x in 0..char_w {
                let px = off_x + x;
                let py = y;
                
                let cx = x as f32 - 7.5;
                let cy = y as f32 - 7.5;
                
                let val = match i {
                    0 => 0, // Space
                    1 => if x == 7 && y == 7 { 255 } else { 0 }, // Dot
                    2 => if x == 7 && y % 4 == 0 { 255 } else { 0 }, // Dotted vertical
                    3 => if cy.abs() < 1.0 || cx.abs() < 1.0 { 255 } else { 0 }, // Plus
                    4 => if (cx - cy).abs() < 1.0 || (cx + cy).abs() < 1.0 { 255 } else { 0 }, // X
                    5 => if x == 0 || x == 15 || y == 0 || y == 15 { 255 } else { 0 }, // Box border
                    6 => if (x+y)%4 == 0 { 255 } else { 0 }, // Diagonals
                    7 => if x%4 == 0 { 255 } else { 0 }, // Vertical lines
                    8 => if y%4 == 0 { 255 } else { 0 }, // Horizontal lines
                    9 => if (x/2)%2 == 0 && (y/2)%2 == 0 { 255 } else { 0 }, // Checker
                    10 => if (cx*cx + cy*cy).sqrt() < 6.0 && (cx*cx + cy*cy).sqrt() > 4.0 { 255 } else { 0 }, // Circle
                    11 => if (cx*cx + cy*cy).sqrt() < 7.0 { 255 } else { 0 }, // Filled Circle
                    12 => if x%2 == 0 { 255 } else { 0 }, // Dense Vertical
                    13 => if (x+y)%2 == 0 { 255 } else { 0 }, // Dense Checker
                    14 => if x > 2 && x < 14 && y > 2 && y < 14 { 255 } else { 0 }, // Filled Box
                    15 => 255, // Full Block
                    _ => 0
                };
                set_pixel(px, py, val);
            }
        }
    }

    gl.tex_image_2d(TEXTURE_2D, 0, RGBA as i32, 256, 16, 0, RGBA, UNSIGNED_BYTE, Some(&data));
    gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MIN_FILTER, NEAREST as i32);
    gl.tex_parameter_i32(TEXTURE_2D, TEXTURE_MAG_FILTER, NEAREST as i32);
    Ok(tex)
}
