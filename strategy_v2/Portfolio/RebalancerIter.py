from datetime import datetime
from croniter import croniter

class RebalancerIter(object):
    def __init__(self,                  
                 cron_expr:str,
                 iter_step:int=1,
    ):
        '''
            This is the rebalance iterator which control the rebalnce peridos in a date format (e.g. every 1st Friday)
        '''        
        self.cron_expr = cron_expr
        self.iter_step = iter_step
        

    def set_base_date(self, base_date:datetime):        
        self.base_date = base_date
        self.cron = croniter(self.cron_expr, self.base_date)                
        return self

    def get_next(self) -> datetime:        
        for _ in range(self.iter_step):
            self.cur_date = self.cron.get_next(datetime)
        return self.cur_date
    
if __name__ == '__main__':
    iter = RebalancerIter('0 0 * * Fri', iter_step=2)
    iter.set_base_date(datetime.today())
    print(iter.get_next())
    print(iter.get_next())
    print(iter.get_next())
    print(iter.get_next())
    print(iter.get_next())
    print(iter.get_next())
    

