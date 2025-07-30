# modules/storage.py
__version__ = "0.1.3"
import os
import urllib.parse
import tempfile
import shutil
import json
import base64
from datetime import datetime, timezone
from huggingface_hub import login, upload_folder, hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
from modules.constants import HF_API_TOKEN, upload_file_types, model_extensions, image_extensions, audio_extensions, video_extensions, doc_extensions, HF_REPO_ID, SHORTENER_JSON_FILE
from typing import Any, Dict, List, Tuple, Union, Optional

# see storage.md for detailed information about the storage module and its functions.

def generate_permalink(valid_files, base_url_external, permalink_viewer_url="surn-3d-viewer.hf.space"):
    """
    Given a list of valid files, checks if they contain exactly 1 model file and 2 image files.
    Constructs and returns a permalink URL with query parameters if the criteria is met.
    Otherwise, returns None.
    """
    model_link = None
    images_links = []
    audio_links = []
    video_links = []
    doc_links = []
    for f in valid_files:
        filename = os.path.basename(f)
        ext = os.path.splitext(filename)[1].lower()
        if ext in model_extensions:
            if model_link is None:
                model_link = f"{base_url_external}/{filename}"
        elif ext in image_extensions:
            images_links.append(f"{base_url_external}/{filename}")
        elif ext in audio_extensions:
            audio_links.append(f"{base_url_external}/{filename}")
        elif ext in video_extensions:
            video_links.append(f"{base_url_external}/{filename}")
        elif ext in doc_extensions:
            doc_links.append(f"{base_url_external}/{filename}")
    if model_link and len(images_links) == 2:
        # Construct a permalink to the viewer project with query parameters.
        permalink_viewer_url = f"https://{permalink_viewer_url}/"
        params = {"3d": model_link, "hm": images_links[0], "image": images_links[1]}
        query_str = urllib.parse.urlencode(params)
        return f"{permalink_viewer_url}?{query_str}"
    return None

def generate_permalink_from_urls(model_url, hm_url, img_url, permalink_viewer_url="surn-3d-viewer.hf.space"):
    """
    Constructs and returns a permalink URL with query string parameters for the viewer.
    Each parameter is passed separately so that the image positions remain consistent.
    
    Parameters:
        model_url (str): Processed URL for the 3D model.
        hm_url (str): Processed URL for the height map image.
        img_url (str): Processed URL for the main image.
        permalink_viewer_url (str): The base viewer URL.
    
    Returns:
        str: The generated permalink URL.
    """
    import urllib.parse
    params = {"3d": model_url, "hm": hm_url, "image": img_url}
    query_str = urllib.parse.urlencode(params)
    return f"https://{permalink_viewer_url}/?{query_str}"

def upload_files_to_repo(
    files: List[Any],
    repo_id: str,
    folder_name: str,
    create_permalink: bool = False,
    repo_type: str = "dataset",
    permalink_viewer_url: str = "surn-3d-viewer.hf.space"
) -> Union[Dict[str, Any], List[Tuple[Any, str]]]:
    """
    Uploads multiple files to a Hugging Face repository using a batch upload approach via upload_folder.

    Parameters:
        files (list): A list of file paths (str) to upload.
        repo_id (str): The repository ID on Hugging Face for storage, e.g. "Surn/Storage".
        folder_name (str): The subfolder within the repository where files will be saved.
        create_permalink (bool): If True and if exactly three files are uploaded (1 model and 2 images),
                                 returns a single permalink to the project with query parameters.
                                 Otherwise, returns individual permalinks for each file.
        repo_type (str): Repository type ("space", "dataset", etc.). Default is "dataset".
        permalink_viewer_url (str): The base viewer URL.

    Returns:
        Union[Dict[str, Any], List[Tuple[Any, str]]]:
            If create_permalink is True and files match the criteria:
                dict: {
                    "response": <upload response>,
                    "permalink": <full_permalink URL>,
                    "short_permalink": <shortened permalink URL>
                }
            Otherwise:
                list: A list of tuples (response, permalink) for each file.
    """
    # Log in using the HF API token.
    login(token=HF_API_TOKEN) # Corrected from HF_TOKEN to HF_API_TOKEN
    
    valid_files = []
    permalink_short = None
    
    # Ensure folder_name does not have a trailing slash.
    folder_name = folder_name.rstrip("/")
    
    # Filter for valid files based on allowed extensions.
    for f in files:
        file_name = f if isinstance(f, str) else f.name if hasattr(f, "name") else None
        if file_name is None:
            continue
        ext = os.path.splitext(file_name)[1].lower()
        if ext in upload_file_types:
            valid_files.append(f)
    
    if not valid_files:
        # Return a dictionary with None values for permalinks if create_permalink was True
        if create_permalink:
            return {
                "response": "No valid files to upload.",
                "permalink": None,
                "short_permalink": None
            }
        return [] 
    
    # Create a temporary directory; copy valid files directly into it.
    with tempfile.TemporaryDirectory(dir=os.getenv("TMPDIR", "/tmp")) as temp_dir:
        for file_path in valid_files:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(temp_dir, filename)
            shutil.copy(file_path, dest_path)
        
        # Batch upload all files in the temporary folder.
        # Files will be uploaded under the folder (path_in_repo) given by folder_name.
        response = upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=folder_name,
            commit_message="Batch upload files"
        )
    
    # Construct external URLs for each uploaded file.
    base_url_external = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{folder_name}"
    individual_links = []
    for file_path in valid_files:
        filename = os.path.basename(file_path)
        link = f"{base_url_external}/{filename}"
        individual_links.append(link)
    
    # If permalink creation is requested and exactly 3 valid files are provided,
    # try to generate a permalink using generate_permalink().
    if create_permalink: # No need to check len(valid_files) == 3 here, generate_permalink will handle it
        permalink = generate_permalink(valid_files, base_url_external, permalink_viewer_url)
        if permalink:
            status, short_id = gen_full_url(
                full_url=permalink,
                repo_id=HF_REPO_ID, # This comes from constants
                json_file=SHORTENER_JSON_FILE # This comes from constants
            )
            if status in ["created_short", "success_retrieved_short", "exists_match"]:
                permalink_short = f"https://{permalink_viewer_url}/?sid={short_id}"
            else: # Shortening failed or conflict not resolved to a usable short_id
                permalink_short = None 
                print(f"URL shortening status: {status} for {permalink}")

            return {
                "response": response,
                "permalink": permalink,
                "short_permalink": permalink_short
            }
        else: # generate_permalink returned None (criteria not met)
            return {
                "response": response, # Still return upload response
                "permalink": None,
                "short_permalink": None
            }

    # Otherwise, return individual tuples for each file.
    return [(response, link) for link in individual_links]

def _generate_short_id(length=8):
    """Generates a random base64 URL-safe string."""
    return base64.urlsafe_b64encode(os.urandom(length * 2))[:length].decode('utf-8')

def _get_json_from_repo(repo_id, json_file_name, repo_type="dataset"):
    """Downloads and loads the JSON file from the repo. Returns empty list if not found or error."""
    try:
        login(token=HF_API_TOKEN)
        json_path = hf_hub_download(
            repo_id=repo_id,
            filename=json_file_name,
            repo_type=repo_type,
            token=HF_API_TOKEN
        )
        with open(json_path, 'r') as f:
            data = json.load(f)
        os.remove(json_path)
        return data
    except RepositoryNotFoundError:
        print(f"Repository {repo_id} not found.")
        return []
    except EntryNotFoundError:
        print(f"JSON file {json_file_name} not found in {repo_id}. Initializing with empty list.")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {json_file_name}. Returning empty list.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while fetching {json_file_name}: {e}")
        return []

def _get_files_from_repo(repo_id, file_name, repo_type="dataset"):
    """Downloads and loads the file from the repo. File must be in upload_file_types. Returns empty list if not found or error."""
    filename = os.path.basename(file_name)
    ext = os.path.splitext(file_name)[1].lower() 
    if ext not in upload_file_types:
        print(f"File {filename} with extension {ext} is not allowed for upload.")
        return None
    else:
        try:
            login(token=HF_API_TOKEN)
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                repo_type=repo_type,
                token=HF_API_TOKEN
            )
            if not file_path:
                return None
            return file_path
        except RepositoryNotFoundError:
            print(f"Repository {repo_id} not found.")
            return None
        except EntryNotFoundError:
            print(f"file {file_name} not found in {repo_id}. Initializing with empty list.")
            return None
        except Exception as e:
            print(f"Error fetching {file_name} from {repo_id}: {e}")
            return None

def _upload_json_to_repo(data, repo_id, json_file_name, repo_type="dataset"):
    """Uploads the JSON data to the specified file in the repo."""
    try:
        login(token=HF_API_TOKEN)
        api = HfApi()
        # Use a temporary directory specified by TMPDIR or default to system temp
        temp_dir_for_json = os.getenv("TMPDIR", tempfile.gettempdir())
        os.makedirs(temp_dir_for_json, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json", dir=temp_dir_for_json) as tmp_file:
            json.dump(data, tmp_file, indent=2)
            tmp_file_path = tmp_file.name
        
        api.upload_file(
            path_or_fileobj=tmp_file_path,
            path_in_repo=json_file_name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Update {json_file_name}"
        )
        os.remove(tmp_file_path) # Clean up temporary file
        return True
    except Exception as e:
        print(f"Failed to upload {json_file_name} to {repo_id}: {e}")
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path) # Ensure cleanup on error too
        return False

def _find_url_in_json(data, short_url=None, full_url=None):
    """
    Searches the JSON data.
    If short_url is provided, returns the corresponding full_url or None.
    If full_url is provided, returns the corresponding short_url or None.
    """
    if not data: # Handles cases where data might be None or empty
        return None
    if short_url:
        for item in data:
            if item.get("short_url") == short_url:
                return item.get("full_url")
    if full_url:
        for item in data:
            if item.get("full_url") == full_url:
                return item.get("short_url")
    return None

def _add_url_to_json(data, short_url, full_url):
    """Adds a new short_url/full_url pair to the data. Returns updated data."""
    if data is None: 
        data = []
    data.append({"short_url": short_url, "full_url": full_url})
    return data

def gen_full_url(short_url=None, full_url=None, repo_id=None, repo_type="dataset", permalink_viewer_url="surn-3d-viewer.hf.space", json_file="shortener.json"):
    """
    Manages short URLs and their corresponding full URLs in a JSON file stored in a Hugging Face repository.

    - If short_url is provided, attempts to retrieve and return the full_url.
    - If full_url is provided, attempts to retrieve an existing short_url or creates a new one, stores it, and returns the short_url.
    - If both are provided, checks for consistency or creates a new entry.
    - If neither is provided, or repo_id is missing, returns an error status.

    Returns:
        tuple: (status_message, result_url)
               status_message can be "success", "created", "exists", "error", "not_found".
               result_url is the relevant URL (short or full) or None if an error occurs or not found.
    """
    if not repo_id:
        return "error_repo_id_missing", None
    if not short_url and not full_url:
        return "error_no_input", None

    login(token=HF_API_TOKEN) # Ensure login at the beginning
    url_data = _get_json_from_repo(repo_id, json_file, repo_type)

    # Case 1: Only short_url provided (lookup full_url)
    if short_url and not full_url:
        found_full_url = _find_url_in_json(url_data, short_url=short_url)
        return ("success_retrieved_full", found_full_url) if found_full_url else ("not_found_short", None)

    # Case 2: Only full_url provided (lookup or create short_url)
    if full_url and not short_url:
        existing_short_url = _find_url_in_json(url_data, full_url=full_url)
        if existing_short_url:
            return "success_retrieved_short", existing_short_url
        else:
            # Create new short_url
            new_short_id = _generate_short_id()
            url_data = _add_url_to_json(url_data, new_short_id, full_url)
            if _upload_json_to_repo(url_data, repo_id, json_file, repo_type):
                return "created_short", new_short_id 
            else:
                return "error_upload", None

    # Case 3: Both short_url and full_url provided
    if short_url and full_url:
        found_full_for_short = _find_url_in_json(url_data, short_url=short_url)
        found_short_for_full = _find_url_in_json(url_data, full_url=full_url)

        if found_full_for_short == full_url: 
            return "exists_match", short_url 
        if found_full_for_short is not None and found_full_for_short != full_url: 
            return "error_conflict_short_exists_different_full", short_url
        if found_short_for_full is not None and found_short_for_full != short_url:
            return "error_conflict_full_exists_different_short", found_short_for_full
        
        # If short_url is provided and not found, or full_url is provided and not found,
        # or neither is found, then create a new entry with the provided short_url and full_url.
        # This effectively allows specifying a custom short_url if it's not already taken.
        url_data = _add_url_to_json(url_data, short_url, full_url)
        if _upload_json_to_repo(url_data, repo_id, json_file, repo_type):
            return "created_specific_pair", short_url
        else:
            return "error_upload", None
                
    return "error_unhandled_case", None # Should not be reached

def _encrypt_private_key(private_key: str, password: str = None) -> str:
    """
    Basic encryption for private keys. In production, use proper encryption like Fernet.
    
    Note: This is a simplified encryption for demonstration. In production environments,
    use proper encryption libraries like cryptography.fernet.Fernet with secure key derivation.
    
    Args:
        private_key (str): The private key to encrypt
        password (str, optional): Password for encryption. If None, uses a default method.
    
    Returns:
        str: Base64 encoded encrypted private key
    """
    # WARNING: This is a basic XOR encryption for demo purposes only
    # In production, use proper encryption like Fernet from cryptography library
    if not password:
        password = "default_encryption_key"  # In production, use secure key derivation
    
    encrypted_bytes = []
    for i, char in enumerate(private_key):
        encrypted_bytes.append(ord(char) ^ ord(password[i % len(password)]))
    
    encrypted_data = bytes(encrypted_bytes)
    return base64.b64encode(encrypted_data).decode('utf-8')

def _decrypt_private_key(encrypted_private_key: str, password: str = None) -> str:
    """
    Basic decryption for private keys. In production, use proper decryption like Fernet.
    
    Args:
        encrypted_private_key (str): Base64 encoded encrypted private key
        password (str, optional): Password for decryption. If None, uses a default method.
    
    Returns:
        str: Decrypted private key
    """
    # WARNING: This is a basic XOR decryption for demo purposes only
    if not password:
        password = "default_encryption_key"  # In production, use secure key derivation
    
    encrypted_data = base64.b64decode(encrypted_private_key)
    decrypted_chars = []
    for i, byte in enumerate(encrypted_data):
        decrypted_chars.append(chr(byte ^ ord(password[i % len(password)])))
    
    return ''.join(decrypted_chars)

def store_issuer_keypair(issuer_id: str, public_key: str, private_key: str, repo_id: str = None) -> bool:
    """
    Store cryptographic keys for an issuer in the private Hugging Face repository.
    
    **IMPORTANT: This function requires a PRIVATE Hugging Face repository to ensure
    the security of stored private keys. Never use this with public repositories.**
    
    The keys are stored in the following structure:
    keys/issuers/{issuer_id}/
    ├── private_key.json (encrypted)
    └── public_key.json
    
    Args:
        issuer_id (str): Unique identifier for the issuer (e.g., "https://example.edu/issuers/565049")
        public_key (str): Multibase-encoded public key
        private_key (str): Multibase-encoded private key (will be encrypted before storage)
        repo_id (str, optional): Repository ID. If None, uses HF_REPO_ID from constants.
    
    Returns:
        bool: True if keys were stored successfully, False otherwise
    
    Raises:
        ValueError: If issuer_id, public_key, or private_key are empty
        Exception: If repository operations fail
    
    Example:
        >>> public_key = "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
        >>> private_key = "z3u2MQhLnQw7nvJRGJCdKdqfXHV4N7BLKuEGFWnJqsVSdgYv"
        >>> success = store_issuer_keypair("https://example.edu/issuers/565049", public_key, private_key)
        >>> print(f"Keys stored: {success}")
    """
    if not issuer_id or not public_key or not private_key:
        raise ValueError("issuer_id, public_key, and private_key are required")
    
    if not repo_id:
        repo_id = HF_REPO_ID
    
    # Sanitize issuer_id for use as folder name
    safe_issuer_id = issuer_id.replace("https://", "").replace("http://", "").replace("/", "_").replace(":", "_")
    
    try:
        # Encrypt the private key before storage
        encrypted_private_key = _encrypt_private_key(private_key)
        
        # Prepare key data structures
        private_key_data = {
            "issuer_id": issuer_id,
            "encrypted_private_key": encrypted_private_key,
            "key_type": "Ed25519VerificationKey2020",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "encryption_method": "basic_xor"  # In production, use proper encryption
        }
        
        public_key_data = {
            "issuer_id": issuer_id,
            "public_key": public_key,
            "key_type": "Ed25519VerificationKey2020",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Store private key
        private_key_path = f"keys/issuers/{safe_issuer_id}/private_key.json"
        private_key_success = _upload_json_to_repo(private_key_data, repo_id, private_key_path, "dataset")
        
        # Store public key
        public_key_path = f"keys/issuers/{safe_issuer_id}/public_key.json"
        public_key_success = _upload_json_to_repo(public_key_data, repo_id, public_key_path, "dataset")
        
        # Update global verification methods registry
        if private_key_success and public_key_success:
            _update_verification_methods_registry(issuer_id, safe_issuer_id, public_key, repo_id)
        
        return private_key_success and public_key_success
        
    except Exception as e:
        print(f"Error storing issuer keypair for {issuer_id}: {e}")
        return False

def get_issuer_keypair(issuer_id: str, repo_id: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieve stored cryptographic keys for an issuer from the private Hugging Face repository.
    
    **IMPORTANT: This function accesses a PRIVATE Hugging Face repository containing
    encrypted private keys. Ensure proper access control and security measures.**
    
    Args:
        issuer_id (str): Unique identifier for the issuer
        repo_id (str, optional): Repository ID. If None, uses HF_REPO_ID from constants.
    
    Returns:
        Tuple[Optional[str], Optional[str]]: (public_key, private_key) or (None, None) if not found
        
    Raises:
        ValueError: If issuer_id is empty
        Exception: If repository operations fail or decryption fails
    
    Example:
        >>> public_key, private_key = get_issuer_keypair("https://example.edu/issuers/565049")
        >>> if public_key and private_key:
        ...     print("Keys retrieved successfully")
        ... else:
        ...     print("Keys not found")
    """
    if not issuer_id:
        raise ValueError("issuer_id is required")
    
    if not repo_id:
        repo_id = HF_REPO_ID
    
    # Sanitize issuer_id for use as folder name
    safe_issuer_id = issuer_id.replace("https://", "").replace("http://", "").replace("/", "_").replace(":", "_")
    
    try:
        # Retrieve public key
        public_key_path = f"keys/issuers/{safe_issuer_id}/public_key.json"
        public_key_data = _get_json_from_repo(repo_id, public_key_path, "dataset")
        
        # Retrieve private key
        private_key_path = f"keys/issuers/{safe_issuer_id}/private_key.json"
        private_key_data = _get_json_from_repo(repo_id, private_key_path, "dataset")
        
        if not public_key_data or not private_key_data:
            print(f"Keys not found for issuer {issuer_id}")
            return None, None
        
        # Extract and decrypt private key
        encrypted_private_key = private_key_data.get("encrypted_private_key")
        if not encrypted_private_key:
            print(f"No encrypted private key found for issuer {issuer_id}")
            return None, None
        
        decrypted_private_key = _decrypt_private_key(encrypted_private_key)
        public_key = public_key_data.get("public_key")
        
        return public_key, decrypted_private_key
        
    except Exception as e:
        print(f"Error retrieving issuer keypair for {issuer_id}: {e}")
        return None, None

def _update_verification_methods_registry(issuer_id: str, safe_issuer_id: str, public_key: str, repo_id: str):
    """
    Update the global verification methods registry with new issuer public key.
    
    Args:
        issuer_id (str): Original issuer ID
        safe_issuer_id (str): Sanitized issuer ID for file system
        public_key (str): Public key to register
        repo_id (str): Repository ID
    """
    try:
        registry_path = "keys/global/verification_methods.json"
        registry_data = _get_json_from_repo(repo_id, registry_path, "dataset")
        
        if not registry_data:
            registry_data = {"verification_methods": []}
        
        # Check if issuer already exists in registry
        existing_entry = None
        for i, method in enumerate(registry_data.get("verification_methods", [])):
            if method.get("issuer_id") == issuer_id:
                existing_entry = i
                break
        
        # Create new verification method entry
        verification_method = {
            "issuer_id": issuer_id,
            "safe_issuer_id": safe_issuer_id,
            "public_key": public_key,
            "key_type": "Ed25519VerificationKey2020",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        if existing_entry is not None:
            # Update existing entry
            registry_data["verification_methods"][existing_entry] = verification_method
        else:
            # Add new entry
            registry_data["verification_methods"].append(verification_method)
        
        # Upload updated registry
        _upload_json_to_repo(registry_data, repo_id, registry_path, "dataset")
        
    except Exception as e:
        print(f"Error updating verification methods registry: {e}")

def get_verification_methods_registry(repo_id: str = None) -> Dict[str, Any]:
    """
    Retrieve the global verification methods registry.
    
    Args:
        repo_id (str, optional): Repository ID. If None, uses HF_REPO_ID from constants.
    
    Returns:
        Dict[str, Any]: Registry data containing all verification methods
    """
    if not repo_id:
        repo_id = HF_REPO_ID
    
    try:
        registry_path = "keys/global/verification_methods.json"
        registry_data = _get_json_from_repo(repo_id, registry_path, "dataset")
        return registry_data if registry_data else {"verification_methods": []}
    except Exception as e:
        print(f"Error retrieving verification methods registry: {e}")
        return {"verification_methods": []}

def list_issuer_ids(repo_id: str = None) -> List[str]:
    """
    List all issuer IDs that have stored keys in the repository.
    
    Args:
        repo_id (str, optional): Repository ID. If None, uses HF_REPO_ID from constants.
    
    Returns:
        List[str]: List of issuer IDs
    """
    if not repo_id:
        repo_id = HF_REPO_ID
    
    try:
        registry = get_verification_methods_registry(repo_id)
        return [method["issuer_id"] for method in registry.get("verification_methods", [])]
    except Exception as e:
        print(f"Error listing issuer IDs: {e}")
        return []
