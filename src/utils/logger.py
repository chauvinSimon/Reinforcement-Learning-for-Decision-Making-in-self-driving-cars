# Class for log
import logging


class Logger:
    def __init__(self, module_name, logfile_name, log_level):
        """

        :param module_name: [string]
        :param logfile_name: [string]
        :param log_level: [int] in [0 - 4]
        """

        self.module_name = module_name
        self.severity_level_list = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
        self.logfile_name = logfile_name
        self.log_level_str = self.severity_level_list[log_level]
        self.log_level_int = log_level

        # logging configuration
        self.logger = logging.getLogger(self.module_name)  # instantiation
        self.logger.setLevel(self.log_level_str)  # lowest-severity log message a logger will handle
        # specify the layout of log records in the final output - ToDo: add milliseconds as well?
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                           datefmt="%Y-%m-%d %H:%M:%S")
        self.file_handler = logging.FileHandler(self.logfile_name)  # Handlers send the log records
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def log(self, message, severity_level):
        # Report events that occur during normal operation of a program (for status monitoring or fault investigation)
        if severity_level == 0:  # Detailed information, typically of interest only when diagnosing problems
            self.logger.debug(message)
        elif severity_level == 1:  # Confirmation that things are working as expected
            self.logger.info(message)

        # Issue a warning regarding a particular runtime event
        if severity_level == 2:  # An indication that something unexpected happened / indicative of some problem in the
            # near future (e.g. ‘disk space low’). The software is still working as expected
            self.logger.warning(message)

        # Report suppression of an error without raising exception (e.g. error handler in a long-running server process)
        if severity_level == 3:  # Due to a more serious problem, the software hasn't been able to perform some function
            self.logger.error(message)
        if severity_level == 4:  # A serious error, indicating that the program itself may be unable to continue running
            self.logger.fatal(message)


if __name__ == "__main__":
    logger = Logger(
        module_name="example_module",
        logfile_name="example_module.log",
        log_level=0,
        )

    for i in range(5):
        logger.log("test", i)
