
class ErrorCode(Exception):
    error_code_list = [
        {
            "code": "10001",
            "message": "输入因子不正确",
            "solution": "1、请检查是否填写因子；2、请检查策略代码是否正常；3、请检查开始时间和结束时间，是否时间范围过短。",
            "example": ['因子为空']
        },
        {
            "code": "10002",
            "message": "结束时间与起始时间之差大于3年",
            "solution": "算力资源有限，请检查起始时间和结束时间，以确保结束时间与起始时间之差为3年以内。",
            "example": ["开始时间和结束时间之间的时间范围不能超过3年",
                        "回测开始时间和回测结束时间之间的时间范围不能超过3年",
                        "预测开始时间和预测结束时间之间的时间范围不能超过3年"]
        },
        {
            "code": "10003",
            "message": "计算函数在系统中未定义",
            "solution": "请参考pandaAI手册，检查计算函数是否书写有误？",
            "example": ["is not defined"]
        },
        {
            "code": "10004",
            "message": "时间格式不匹配",
            "solution": "请检查时间格式，确保其格式为'%Y%m%d'，如：20250101",
            "example": ["does not match format '%Y%m%d'", "unconverted data remains"]
        },
        {
            "code": "10005",
            "message": "策略代码编译异常",
            "solution": "请检查Python代码语法是否合法",
            "example": ["策略代码编译异常", "Code syntax error"]
        },
        {
            "code": "10006",
            "message": "缺少基础因子或基础因子不正确，如：open、close、high、low等",
            "solution": "请参考pandaAI手册，检查是否输入了正确的基础因子",
            "example": ['Missing required base factors']
        },
        {
            "code": "10007",
            "message": "输入节点和综合因子构建节点的编码方式不一致",
            "solution": "请检查输入节点和综合因子构建节点的编码方式是否一致，如：输入节点为公式输入，而综合因子构建节点的编码方式为python，则会报错。请确保二者的编码方式一致！",
            "example": ["NoneType' object has no attribute 'reset_index'", "Error in formula", "Factor class load failed"]
        },
        {
            "code": "10008",
            "message": "特征公式不正确",
            "solution": "请检查特征公式是否正确，如：括号没有闭合、括号写成中文、特征公式不存在",
            "example": ["SyntaxError", "SyntaxError: invalid character"]
        },
        {
            "code": "10009",
            "message": "特征工程构建时标签公式不正确",
            "solution": "请检查特征公式是否正确，如：括号没有闭合、括号写成中文、特征公式不存在",
            "example": ["标签不能为空", "FeatureModel 的标签不能为空"]
        }
    ]

    # 获取错误信息
    @classmethod
    def get_error_message(cls, code):
        for error in cls.error_code_list:
            if error['code'] == code:
                return error
        return None

    # 根椐错误示例查询对应的错误编码、错误消息及解决方案
    @classmethod
    def get_error_by_message(cls, error_msg):
        for error in cls.error_code_list:
            # 遍历错误示例，如果错误消息中包含错误示例，则返回对应的错误代码、错误消息及解决方案
            for example in error['example']:
                if example in error_msg or example == error_msg:
                    return f"错误代码：{error['code']} \n错误消息：{error['message']} \n解决方案：{error['solution']}"
        return ""