import sys
import os

def concat_folder(folder, element):
    """folder and element are strings"""
    if folder[-1] == '/':
        return folder + element
    return folder + '/' + element

def merge(from_dir, to_dir):
    assert os.path.exists(from_dir)
    assert os.path.exists(to_dir)
    for subfolder in os.listdir(from_dir):
        from_subfolder = concat_folder(from_dir, subfolder)
        to_subfolder = concat_folder(to_dir, subfolder)
        assert os.path.exists(to_subfolder)
        for filename in os.listdir(from_subfolder):
            if filename == 'flags.pickle':
                continue
            from_file = concat_folder(from_subfolder, filename)
            to_file = concat_folder(to_subfolder, filename)
            assert not os.path.exists(to_file), to_file
            os.rename(from_file, to_file)

if __name__ == '__main__':
    from_dir, to_dir = sys.argv[1:]
    merge(from_dir, to_dir)
