import importlib
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.schedulers.background import BlockingScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED
from utils.log.log_factory import LogFactory


class CrontabManager(object):
    _scheduler = None
    _logger = LogFactory.get_logger()
    __job_dict = dict()

    def __init__(self):
        pass

    @classmethod
    def init_cron_manager(cls):
        job_defaults = {
            'misfire_grace_time': 600,
            'coalesce': False,
            'max_instances': 5
        }

        executors = {
            'default': ThreadPoolExecutor(8),
            'processpool': ProcessPoolExecutor(4)
        }

        cls._scheduler = BlockingScheduler(
            executors=executors,
            job_defaults=job_defaults,
            timezone='Asia/Shanghai')
        cls._scheduler.add_listener(cls.my_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)

    @classmethod
    def my_listener(cls, event):
        if event.code == EVENT_JOB_ERROR:
            task_name = CrontabManager.__job_dict[event.job_id]
            CrontabManager._logger.error('任务执行异常，任务：%s,异常信息：%s' % (task_name, str(event.exception)))
        elif event.code == EVENT_JOB_MISSED:
            task_name = CrontabManager.__job_dict[event.job_id]
            CrontabManager._logger.error('任务执行丢失，任务：%s' % task_name)
        else:
            pass
            # print('The job worked')

    @classmethod
    def add_job(cls, method, cron_type, task_name, **arg):
        """
        添加任务
        :param method: 调度的方法
        :param type:   调度器类型（cron; ）
        :param arg:
        :return:
        """
        job = cls._scheduler.add_job(method, cron_type, replace_existing=True, **arg)

        print("add job %s successful!  next_run_time: " % str(job))
        CrontabManager.__job_dict[job.id] = task_name
        return job.id

    @classmethod
    def stop_job(cls, job_id):
        cls._scheduler.pause_job(job_id)

    @classmethod
    def remove_job(cls, job_id):
        cls._scheduler.remove_job(job_id)

    @classmethod
    def resume_job(cls, job_id):
        cls._scheduler.resume_job(job_id)

    @classmethod
    def stop_all_job(cls):
        cls._scheduler.pause()

    @classmethod
    def start_all_job(cls):
        cls._scheduler.resume()

    @classmethod
    def start_scheduler(cls):
        cls._scheduler.start()

    @staticmethod
    def init_all_task(module_list):
        for module in module_list:
            imp_module = module
            ip_module = importlib.import_module(imp_module, '')
            for attr in dir(ip_module):
                func = getattr(ip_module, attr)
                if hasattr(func, 'decorator_name'):
                    decorator_name = func.decorator_name
                    if decorator_name == 'cron_task':
                        try:
                            func()
                            # print('定时任务添加成功，任务：%s' % func.task_name)
                            CrontabManager._logger.info('定时任务添加成功，任务：%s' % func.task_name)
                        except Exception as e:
                            CrontabManager._logger.info('定时任务添加失败，任务：%s,异常信息：%s' % (str(func.task_name), str(e)))
                            # print('定时任务添加失败，任务：%s,异常信息：%s' % (str(func.task_name), str(e)))
