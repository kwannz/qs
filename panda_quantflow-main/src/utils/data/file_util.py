import os


class FileUtil(object):

    @classmethod
    def get_files(cls, file_dir, file_name_end=None):
        file_all = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if file_name_end:
                    if file.endswith(file_name_end):
                        file_all.append(os.path.join(root, file))
                else:
                    file_all.append(os.path.join(root, file))
        return file_all
