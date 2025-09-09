from fastapi import HTTPException, status
from panda_plugins.base.work_node_registery import ALL_WORK_NODES
from panda_server.models.all_plugins_response import AllPluginsResponse, PluginInfo, PluginGroup
import traceback
import logging

# 定义 collection 名称
COLLECTION_NAME = "plugins"

# 获取 logger
logger = logging.getLogger(__name__)

async def get_all_plugins_logic():
    all_plugins = []

    for name, node_class in ALL_WORK_NODES.items():
        # 获取节点基本信息
        plugin_info = PluginInfo(
            object_type = "plugin",
            name=getattr(node_class, "__work_node_name__"),
            display_name=getattr(node_class, "__work_node_display_name__"),
            group=getattr(node_class, "__work_node_group__"),
            type=getattr(node_class, "__work_node_type__"),
            short_description=getattr(node_class, "__short_description__"),
            long_description=getattr(node_class,"__long_description__"),
            box_color=getattr(node_class, "__work_node_box_color__"),
        )

        # 获取输入模型的JSON Schema
        try:
            input_model = node_class.input_model()
            if input_model:
                plugin_info.input_schema = input_model.model_json_schema()
        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(
                f"unexpected error in get input schema of work node {name} \nerror: {e}\n{stack_trace}"
            )
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 获取输出模型的JSON Schema
        try:
            output_model = node_class.output_model()
            if output_model:
                plugin_info.output_schema = output_model.model_json_schema()
        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(
                f"unexpected error in get output schema of work node {name} \nerror: {e}\n{stack_trace}"
            )

        all_plugins.append(plugin_info)
    
    """
    整理 all_plugins 构建层级结构
    1. 根据 "/" 分割 group 为多个层级
    2. 构建嵌套的树结构
    3. 将插件添加到最终的层级中
    """
    group_tree = {}
    
    for plugin in all_plugins:
        # 根据 "/" 分割 group 为多个层级
        group_parts = plugin.group.split('/')
        current_level = group_tree
        
        # 构建嵌套的树结构
        for part in group_parts:
            if part not in current_level:
                current_level[part] = {'subgroups': {}, 'plugins': []}
            current_level = current_level[part]['subgroups']
        
        # 将插件添加到最终的层级中
        # 回退到正确的位置
        current_level = group_tree
        for part in group_parts[:-1]:
            current_level = current_level[part]['subgroups']
        
        if group_parts:
            current_level[group_parts[-1]]['plugins'].append(plugin)

    formatted_plugins = build_nested_structure(group_tree)
    
    return AllPluginsResponse(data=formatted_plugins)



def build_nested_structure(tree_node):
    """
    构建嵌套的层级结构
    1. 根据 group 的字符串排序，决定每个 group 的先后顺序
    2. 每个 group 内，优先排列子group，而后才是 plugin
    3. plugin 也按照 display_name 的字符串排序
    """
    result = []
    
    # 根据 group 的字符串排序，决定每个 group 的先后顺序
    sorted_groups = sorted(tree_node.keys())
    
    for group_name in sorted_groups:
        group_data = tree_node[group_name]
        
        # 构建children列表
        children = []
        
        # 每个 group 内，优先排列子group，而后才是 plugin
        # 先添加子组
        if group_data['subgroups']:
            subgroup_items = build_nested_structure(group_data['subgroups'])
            children.extend(subgroup_items)
        
        # plugin 也按照 display_name 的字符串排序
        if group_data['plugins']:
            sorted_plugins = sorted(group_data['plugins'], key=lambda p: p.display_name)
            children.extend(sorted_plugins)
        
        # 使用 PluginGroup 模型创建组对象
        group_item = PluginGroup(
            name=group_name,
            children=children
        )
        
        result.append(group_item)
    
    return result