
import pyaudio
p = pyaudio.PyAudio()
print('--- Audio Devices ---')
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    name = d.get('name', 'Unknown')
    inputs = d.get('maxInputChannels', 0)
    rate = d.get('defaultSampleRate', 0)
    host_api = d.get('hostApi', -1)
    print(f'[{i}] {name} (Inputs: {inputs}, Rate: {rate}, HostAPI: {host_api})')

print('\n--- Host APIs ---')
wasapi_api_idx = -1
for i in range(p.get_host_api_count()):
    api = p.get_host_api_info_by_index(i)
    print(f'[{i}] {api["name"]} (Type: {api["type"]}, DefaultOut: {api["defaultOutputDevice"]})')
    if api["type"] == 13:
        wasapi_api_idx = i

if wasapi_api_idx != -1:
    print(f"\n--- WASAPI Specific Check (API {wasapi_api_idx}) ---")
    api_info = p.get_host_api_info_by_index(wasapi_api_idx)
    def_out = api_info.get("defaultOutputDevice", -1)
    print(f"Default WASAPI output device: {def_out}")

from audio_ingest import AudioIngest
try:
    print("\n--- AudioIngest Init Log ---")
    ai = AudioIngest()
    print('\nSuccessfully initialized AudioIngest.')
except Exception as e:
    import traceback
    print(f'\nAudioIngest Fatal Fail: {e}')
    traceback.print_exc()
