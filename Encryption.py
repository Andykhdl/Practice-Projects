import os
import pyAesCrypt
from zipfile import ZipFile

def encrypt_folder(folder_path, password):
    # Create a ZIP archive of the folder
    zip_path = folder_path + ".zip"
    with ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

    # Encrypt the ZIP file
    encrypted_zip_path = zip_path + ".aes"
    pyAesCrypt.encryptFile(zip_path, encrypted_zip_path, password)
    return encrypted_zip_path

# Example usage
folder_to_encrypt = ""
password = ""
encrypted_folder = encrypt_folder(folder_to_encrypt, password)
print(f"Folder encrypted and saved as: {encrypted_folder}")