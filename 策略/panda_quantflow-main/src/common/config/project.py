from configparser import ConfigParser
import os
class ProjectConfig:
    _project_config_dict = dict()

    @staticmethod
    def get_config_parser(root_path):
        if root_path in ProjectConfig._project_config_dict.keys():
            return ProjectConfig._project_config_dict[root_path]
        else:
            # 获取全局
            all_config_filename = 'config.ini'
            all_file_path = os.path.join(root_path, all_config_filename)
            all_config_parser = ConfigParser()
            all_config_parser.read(all_file_path)
            config_file = all_config_parser.get('config', 'config_file')
            config_filename = ("config_%s.ini" % config_file)
            file_path = os.path.join(root_path, config_filename)
            _config_parser = ConfigParser()
            _config_parser.read(file_path, encoding='utf-8')
            ProjectConfig._project_config_dict[root_path] = _config_parser
            return _config_parser