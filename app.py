# app.py
import streamlit as st
import tiktoken
from pathlib import Path
import chardet
import zipfile
import tempfile
import shutil
import time

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Code Extractor for LLMs (ZIP Upload)")

# --- Helper Functions ---

def detect_encoding(file_path):
    """Detects the encoding of a file."""
    try:
        with open(file_path, 'rb') as f:
            # Read a small chunk; increase if needed for better detection on some files
            raw_data = f.read(5000)
            result = chardet.detect(raw_data)
            # Provide a fallback encoding if detection confidence is low or encoding is None
            return result['encoding'] if result and result['encoding'] and result['confidence'] > 0.5 else 'utf-8'
    except FileNotFoundError:
        # Handle case where file might disappear between listing and reading (less likely in temp dir)
        st.warning(f"File not found during encoding detection: {file_path}")
        return 'utf-8'
    except Exception as e:
        st.warning(f"Error detecting encoding for {file_path}, defaulting to utf-8: {e}")
        return 'utf-8'

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Calculates the token count for a given text and model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except ModuleNotFoundError:
        st.error("Tiktoken library not found. Please install it: pip install tiktoken")
        return 0
    except Exception as e:
        # Provide more specific error for missing encoding info
        if "Could not find encoding" in str(e):
             st.error(f"Tiktoken error: Could not find encoding for model '{model}'. Check model name.")
        else:
             st.error(f"Could not count tokens for model '{model}': {e}. Using approximation (chars/4).")
        # Fallback simple approximation
        return len(text) // 4

def get_files(root_dir: Path, extensions: list[str], exclusions_lower: list[str]) -> tuple[list[Path], Path]:
    """Recursively finds files matching extensions, avoiding exclusions.
       Returns tuple: (matched_files, scan_root)
    """
    matched_files = []
    if not root_dir.is_dir():
        st.error(f"Error: Invalid processing directory '{root_dir}'.")
        return [], root_dir # Return original root if invalid

    valid_extensions = [f".{ext.lstrip('.').lower()}" for ext in extensions]

    # Determine the effective root directory for scanning (handles single folder in zip)
    scan_root = root_dir
    try:
        items_in_root = list(root_dir.iterdir())
        # If only one item exists and it's a directory, and not in exclusions, assume it's the project root
        if len(items_in_root) == 1 and items_in_root[0].is_dir():
            potential_root_name = items_in_root[0].name.lower()
            # Check against normalized exclusion names
            if potential_root_name not in exclusions_lower:
                st.info(f"Detected single root folder '{items_in_root[0].name}' in ZIP, scanning inside it.")
                scan_root = items_in_root[0]
            else:
                 st.warning(f"Single item '{items_in_root[0].name}' found in ZIP root, but it matches an exclusion rule. Scanning from ZIP root.")

    except Exception as e:
        st.warning(f"Could not determine single root folder, scanning from ZIP root: {e}")
        scan_root = root_dir # Fallback safely

    st.write(f"Scanning for files within: {scan_root.relative_to(root_dir) if scan_root != root_dir else '.'}") # Show relative scan path

    # Use rglob on the determined scan_root
    for item in scan_root.rglob('*'):
        excluded = False
        try:
            # 1. Check if the item itself is excluded by name
            if item.name.lower() in exclusions_lower:
                excluded = True

            # 2. Check if any parent directory *within the scan_root* is excluded
            if not excluded:
                current_parent = item.parent
                # Traverse upwards until we hit the scan_root or the top level
                while current_parent != scan_root and current_parent != current_parent.parent:
                    if current_parent.name.lower() in exclusions_lower:
                        excluded = True
                        break
                    current_parent = current_parent.parent

            if excluded:
                # Use continue to skip adding this item and processing its children implicitly via rglob skipping
                continue

            # 3. If not excluded and is a file matching the extension, add it
            if item.is_file() and item.suffix.lower() in valid_extensions:
                # Store the absolute path for reading
                matched_files.append(item.resolve()) # Ensure path is absolute

        except PermissionError:
            st.warning(f"Skipping due to permission error: {item.relative_to(scan_root)}")
        except FileNotFoundError:
             st.warning(f"Skipping file/directory that seems to have disappeared: {item}")
        except Exception as e:
            # Generate relative path for warning message if possible
            try:
                rel_path_warn = item.relative_to(scan_root)
            except ValueError:
                rel_path_warn = item # Fallback to absolute if not under scan_root somehow
            st.warning(f"Skipping due to unexpected error processing {rel_path_warn}: {e}")

    return matched_files, scan_root # Return the list and the actual root used for scanning


def read_and_combine(files: list[Path], relative_to_dir: Path) -> tuple[str, int]:
    """Reads files and combines their content with headers relative to a specific directory.
       Returns tuple: (combined_text, files_processed_count)
    """
    combined_content = []
    total_files_to_attempt = len(files)
    files_processed_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    errors_encountered = []

    status_text.text(f"Preparing to read {total_files_to_attempt} files...")
    time.sleep(0.1) # Brief pause for UI update

    for i, file_path in enumerate(files):
        try:
            # Ensure file_path is absolute before making relative
            absolute_file_path = file_path.resolve()
            relative_path = absolute_file_path.relative_to(relative_to_dir)
            status_text.text(f"Processing file {i+1}/{total_files_to_attempt}: {relative_path}")

            header = f"\n{'=' * 10} File: {relative_path} {'=' * 10}\n\n"
            combined_content.append(header)

            # Detect encoding before reading
            encoding = detect_encoding(absolute_file_path)

            try:
                 with open(absolute_file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                 combined_content.append(content)
                 combined_content.append("\n") # Add newline after content
                 files_processed_count += 1
            except UnicodeDecodeError:
                 st.warning(f"Could not decode file {relative_path} with detected encoding '{encoding}'. Trying 'latin-1'.")
                 try:
                    # Try reading with latin-1 as a common fallback
                    with open(absolute_file_path, 'r', encoding='latin-1', errors='replace') as f:
                        content = f.read()
                    combined_content.append(content)
                    combined_content.append("\n")
                    files_processed_count += 1 # Count as processed even with fallback
                 except Exception as e_inner:
                    error_msg = f"Failed to read file {relative_path} even with fallback encoding 'latin-1': {e_inner}"
                    st.error(error_msg)
                    errors_encountered.append(error_msg)
                    combined_content.append(f"[Error reading file: {e_inner}]\n") # Add error marker
            except FileNotFoundError:
                error_msg = f"File not found during read attempt: {relative_path}"
                st.error(error_msg)
                errors_encountered.append(error_msg)
                combined_content.append(f"[Error: File not found]\n")
            except PermissionError:
                 error_msg = f"Permission denied reading file: {relative_path}"
                 st.error(error_msg)
                 errors_encountered.append(error_msg)
                 combined_content.append(f"[Error: Permission denied]\n")
            except Exception as e_read:
                error_msg = f"Could not read file {relative_path}: {e_read}"
                st.error(error_msg)
                errors_encountered.append(error_msg)
                combined_content.append(f"[Error reading file: {e_read}]\n") # Add error marker

        except ValueError as ve:
            # Handle cases where relative_to calculation fails (should be rare with resolve)
             error_msg = f"Error creating relative path for {file_path} against {relative_to_dir}: {ve}"
             st.error(error_msg)
             errors_encountered.append(error_msg)
             combined_content.append(f"\n{'=' * 10} Error processing path: {file_path.name} {'=' * 10}\n[Error: {ve}]\n")
        except Exception as e:
            # Catch-all for other unexpected issues during the loop for a single file
            error_msg = f"Unexpected error processing file entry {file_path.name}: {e}"
            st.error(error_msg)
            errors_encountered.append(error_msg)
            combined_content.append(f"\n{'=' * 10} Error processing entry: {file_path.name} {'=' * 10}\n[Error: {e}]\n")

        # Update progress bar regardless of success/failure for this file
        progress_bar.progress((i + 1) / total_files_to_attempt)

    status_text.text(f"Finished reading files. Successfully processed {files_processed_count} out of {total_files_to_attempt} found files.")
    if errors_encountered:
        st.warning(f"Encountered {len(errors_encountered)} errors during file reading. See messages above.")
    progress_bar.empty() # Remove progress bar after completion
    return "".join(combined_content), files_processed_count


# --- Streamlit App UI ---

st.title("📂 Codebase Extractor for LLMs (ZIP Upload)")
st.markdown("""
1.  **ZIP your project folder.** Make sure the `.zip` file directly contains your project files/folders (not nested inside another folder *within* the zip, if possible).
2.  Upload the ZIP file below.
3.  Configure options like included file types and exclusions.
4.  Click 'Process ZIP File' to extract the code and count tokens.
""")

# --- Inputs ---
st.header("Configuration")

# 1. ZIP File Uploader
uploaded_zip = st.file_uploader(
    "Upload your project folder as a ZIP file",
    type=["zip"],
    key="zip_uploader",
    accept_multiple_files=False # Ensure only one file is uploaded
)

# 2. File Extensions Input
default_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.scss', '.md', '.json', '.yaml', '.yml', '.txt', '.sh', 'Dockerfile', '.env.example']
selected_extensions = st.multiselect(
    "Select file extensions to include:",
    options=sorted(list(set(default_extensions + ['.java', '.cs', '.go', '.php', '.rb', '.swift', '.kt', '.c', '.cpp', '.h', '.hpp', '.xml', '.sql', '.gradle', '.properties']))), # Added more common types and sorted
    default=default_extensions,
    key="extensions"
)

# 3. Exclusions Input
default_exclusions = ".git\n.vscode\n.idea\nnode_modules\nvenv\n.venv\nenv\n__pycache__\ndist\nbuild\ntarget\n*.log\n*.lock\n*.env\n.DS_Store\npackage-lock.json\nyarn.lock\ncomposer.lock"
exclusions_str = st.text_area(
    "Enter directory or file names to exclude (one per line, case-insensitive):",
    value=default_exclusions,
    height=150,
    key="exclusions", # This key links the widget to st.session_state.exclusions
    help="Matches exact names (case-insensitive) anywhere within the extracted project structure. Excludes files/folders with these names."
)
# !! Removed the problematic line: st.session_state['exclusions'] = exclusions_str

# 4. Tokenizer Model Input
model_options = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "text-embedding-ada-002", "text-davinci-003"] # Added one more
selected_model = st.selectbox(
    "Select LLM model for token counting:",
    options=model_options,
    index=0, # Default to gpt-4
    key="model"
)

# --- Processing ---
st.header("Process & Output")

# Only show the button if a file has been uploaded
if uploaded_zip is not None:
    if st.button("🚀 Process ZIP File & Count Tokens", key="process_button", type="primary"):
        if not selected_extensions:
             st.warning("⚠️ Please select at least one file extension to include.")
        else:
            # Create the exclusions list *here* using the current value from the text_area
            # Accessing exclusions_str gets the latest value entered by the user
            exclusions_lower = [line.strip().lower() for line in exclusions_str.splitlines() if line.strip()]

            # Use a temporary directory that gets automatically cleaned up
            # Using 'with' ensures cleanup even if errors occur
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir).resolve() # Ensure it's an absolute path
                extracted = False
                zip_file_name = uploaded_zip.name

                # 1. Extract the ZIP file
                try:
                    progress_text = st.empty() # Placeholder for extraction progress text
                    progress_text.text(f"Extracting '{zip_file_name}'...")
                    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir_path)
                    extracted = True
                    progress_text.success(f"✅ Successfully extracted '{zip_file_name}' to temporary location.")
                    time.sleep(1) # Give user time to see success message
                    progress_text.empty() # Clear the extraction message

                except zipfile.BadZipFile:
                    st.error(f"❌ Error: Uploaded file '{zip_file_name}' is not a valid ZIP file or is corrupted.")
                except Exception as e:
                    st.error(f"❌ Error extracting ZIP file: {e}")
                    # Potentially log more details here if needed in a real deployment
                    # print(f"Extraction error: {e}") # For server-side debugging

                if extracted:
                    # The root for scanning initially is the temporary directory path
                    initial_scan_dir = temp_dir_path

                    # 2. Find relevant files within the extracted structure
                    # Pass the prepared lowercase exclusions list
                    files_to_process, scan_root_path = get_files(initial_scan_dir, selected_extensions, exclusions_lower)
                    # 'scan_root_path' is the actual directory where scanning started (handles single nested folder)

                    if not files_to_process:
                        st.warning(f"⚠️ No files found matching the criteria within '{zip_file_name}'. Check extensions and exclusions.")
                    else:
                        st.success(f"🔍 Found {len(files_to_process)} files matching criteria.")

                        # 3. Read and combine content, making paths relative to the scan root
                        combined_text, files_read_count = read_and_combine(files_to_process, scan_root_path) # Pass the correct relative root

                        # 4. Count tokens
                        if combined_text: # Only count tokens if there's text
                            with st.spinner(f"Counting tokens using '{selected_model}' tokenizer..."):
                                token_count = count_tokens(combined_text, model=selected_model)
                        else:
                            token_count = 0
                            st.warning("No text content was extracted to count tokens.")


                        # 5. Display results
                        st.subheader("📊 Results")
                        # Use columns for better layout
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Files Found", f"{len(files_to_process)}")
                        col2.metric("Files Read", f"{files_read_count}")
                        col3.metric(f"Tokens ({selected_model})", f"{token_count:,}" if token_count > 0 else "0")


                        st.subheader("📋 Combined Code Content")
                        st.text_area(
                            "Copy the text below and paste it into your LLM prompt:",
                            value=combined_text if combined_text else "No content extracted based on criteria.",
                            height=600, # Adjust height as needed
                            key="combined_output",
                            disabled=(not combined_text) # Disable if no content
                        )

                        # Add a download button only if there is content
                        if combined_text:
                            try:
                                # Sanitize zip filename for the output txt filename
                                safe_filename_base = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in Path(zip_file_name).stem)
                                download_filename = f"{safe_filename_base}_code_extract.txt"
                            except Exception:
                                download_filename = "code_extract.txt" # Fallback

                            st.download_button(
                                label="📥 Download Combined Text (.txt)",
                                data=combined_text.encode('utf-8'), # Encode to bytes for download
                                file_name=download_filename,
                                mime="text/plain"
                            )

            # End of 'with tempfile.TemporaryDirectory()'
            st.info("Temporary extraction directory has been cleaned up.") # Let user know cleanup happened

else:
    st.info("⬆️ Upload a ZIP file containing your project to begin.")


# --- Footer ---
st.markdown("---")
st.markdown("Developed with Streamlit & Python.")