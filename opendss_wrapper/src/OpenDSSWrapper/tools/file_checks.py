import os


def check_files_exist(path, file_names):
    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            return False
    return True


def find_missing_files(path, file_names):
    missing_file_names = []
    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            missing_file_names.append(file_name)
    return missing_file_names


def check_strings_in_file_names(path, strings_to_find):
    file_names = os.listdir(path)

    for string in strings_to_find:
        string_found = False
        for file_name in file_names:
            if string in file_name:
                string_found = True
                break
        if not string_found:
            return False

    return True


def find_missing_strings_in_file_names(path, strings_to_find):
    file_names = os.listdir(path)
    missing_strings = []

    for string in strings_to_find:
        string_found = False
        for file_name in file_names:
            if string in file_name:
                string_found = True
                break
        if not string_found:
            missing_strings.append(string)

    return missing_strings
