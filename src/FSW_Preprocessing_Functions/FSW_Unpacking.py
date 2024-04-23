def setup_data_directory(origin_dir_path):
    """
    Setups the data directory by copying and unzipping the specified data.

    :param origin_dir_path: Path to the original directory containing the data to be copied and unzipped.
    :type origin_dir_path: str
    """
    # Convert the origin_dir_path to a Path object if it's not already one
    origin_dir_path = Path(origin_dir_path)

    # Creates a working directory where the data will be unzipped
    data_dir_path = Path('./data')
    if os.path.exists(data_dir_path):
        print(f"{data_dir_path} directory already exists")
    else:
        data_dir_path.mkdir(parents=True, exist_ok=True)

    # Copies the data into the working directory
    try:
        hx_training_dest_dir_path = os.path.join(data_dir_path, "hx_training")
        shutil.copytree(origin_dir_path, hx_training_dest_dir_path)
        print("Data copied successfully.")
    except FileExistsError:
        print("Data already copied.")

    # Unzips the data that was copied before into the working directory
    try:
        zipfile_path = os.path.join(hx_training_dest_dir_path, "hx_training_classify.zip")
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir_path)
        print("Data unzipped successfully.")
    except FileNotFoundError:
        print("Zip file not found.")
    except Exception as e:
        print(f"An error occurred while unzipping: {e}")
