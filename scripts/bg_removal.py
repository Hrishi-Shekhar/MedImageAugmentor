from rembg import remove
import os

def remove_bg_batch(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_no_bg.png")
            
            with open(input_path, 'rb') as input_file:
                input_data = input_file.read()
                output_data = remove(input_data)
                with open(output_path, 'wb') as output_file:
                    output_file.write(output_data)
            
            print(f"Processed: {filename}")

