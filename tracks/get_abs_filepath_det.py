import os


def get_detection_folders(parent_dir):
    """
    Get a list of absolute paths of folders in the given parent directory
    where the last word (separated by underscores) is 'detection'.

    Args:
        parent_dir (str): The root directory to search.

    Returns:
        list: A list of absolute paths of matching folders.
    """
    detection_folders = []
    for root, dirs, files in os.walk(parent_dir):
        for dir_name in dirs:
            # Split the folder name by `_` and check the last word
            if dir_name.split('_')[-1].lower() == 'detection':
                path = os.path.abspath(os.path.join(root, dir_name))
                detection_folders.append(path.replace("\\", "/"))
    return detection_folders


# Example usage
if __name__ == "__main__":
    parent_directory = input("Enter the parent directory path: ")
    if os.path.isdir(parent_directory):
        result = get_detection_folders(parent_directory)
        print("Folders with 'detection' as the last word:")
        for path in result:
            print(f"'{path}'")
    else:
        print("The provided path is not a valid directory.")
