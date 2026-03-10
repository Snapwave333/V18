# Use the official Rust image as a base
FROM rust:1-slim as builder

# Create a new non-root user
RUN useradd --user-group --create-home --shell /bin/false app

# Set the working directory
WORKDIR /app

# Copy the Cargo.toml and Cargo.lock files
COPY Cargo.toml Cargo.lock ./

# Create a dummy src directory to cache dependencies
RUN mkdir src
RUN echo "fn main() {}" > src/main.rs
RUN cargo build --release

# Copy the actual source code
COPY src ./src

# Build the application
RUN cargo build --release

# --- Final Stage ---
FROM debian:buster-slim

# Copy the non-root user from the builder stage
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group

# Copy the application binary from the builder stage
COPY --from=builder /app/target/release/rust-vj-backend /usr/local/bin/rust-vj-backend

# Copy the templates directory
COPY templates ./templates

# Use the non-root user
USER app

# Expose the port the app runs on
EXPOSE 5001

# Run the binary
CMD ["rust-vj-backend"]
