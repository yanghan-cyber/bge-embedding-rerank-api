class ModelNotLoadedError(Exception):
    """当模型未加载时引发的自定义异常"""
    def __init__(self, message="No model has been loaded"):
        self.message = message
        super().__init__(self.message)
