import sqlite3

class NNIDatabase(object):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        pass
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
    def get_best_params(self):
        self.cursor.execute("SELECT * FROM MetricData")
        all_data_record = self.cursor.fetchall()

        # find the minimum data record (sort by the last column)
        all_data_record.sort(key=lambda x: float(x[-1].replace('"','')))
        lowest_metric_data_record = all_data_record[0]
        best_params = {}
        if lowest_metric_data_record:
            print(f"Lowest metric data record: {lowest_metric_data_record}")
            trial_job_id = lowest_metric_data_record[1]
            self.cursor.execute("SELECT * FROM TrialJobEvent WHERE trialJobId = ?", (trial_job_id,))
            trial_job_event_records = self.cursor.fetchall()
            is_trail_success = False
            for record in trial_job_event_records:
                if record[2] == "SUCCEEDED":
                    is_trail_success = True
                    break
            for record in trial_job_event_records:
                if record[2] == "WAITING":
                    best_params = eval(record[3])['parameters']
        else:
            print("No metric data records found.")
        return best_params
    
    def close(self):
        self.conn.close()

if __name__ == '__main__':
    db_path = ".nnidatabase/gemm_3bit_16x3072x768/db/nni.sqlite" 
    db = NNIDatabase(db_path)
    print(db.connect())
    print(db.get_best_params())
    print(db.close())