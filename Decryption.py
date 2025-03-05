import os
import pyAesCrypt
from zipfile import ZipFile

def decrypt_folder(encrypted_zip_path, output_folder, password):
    # Decrypt the ZIP file
    decrypted_zip_path = encrypted_zip_path[:-4]  # Remove .aes extension
    pyAesCrypt.decryptFile(encrypted_zip_path, decrypted_zip_path, password, 64*1024*1024)

    # Extract the ZIP file
    with ZipFile(decrypted_zip_path, 'r') as zipf:
        zipf.extractall(output_folder)

# Example usage
encrypted_zip_path = ""
password = ""
output_folder = ""
decrypt_folder(encrypted_zip_path, output_folder, password )
print(f"Folder decrypted and saved to: {output_folder}")