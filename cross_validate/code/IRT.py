import numpy
import os
import json
class IRT:
    def __init__(self,score_matrix:numpy.ndarray, model_type:str,temp_path:str):
        assert os.path.exists(temp_path)
        self.ability = []
        self.diff = []
        self.disc = []
        self.temp_path = temp_path
        self.score_matrix = score_matrix
        self.model_type = model_type 
    def fit(self):
        self.delete_all_files_in_dir(self.temp_path)
        score_path = self.save_to_input_file()
        command = "py-irt train {model_type} {input_file} {output_path} --lr 0.02 --epochs 100"
        os.system(command.format(model_type=self.model_type,input_file=score_path,output_path=self.temp_path))
        attribute = self.read_attribute(os.path.join(self.temp_path,"best_parameters.json"))
        return attribute


    def delete_all_files_in_dir(self, directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


    def save_to_input_file(self):
        score_path = os.path.join(self.temp_path,"score.jsonl")
        with open(score_path, 'w') as f:
            for i, row in enumerate(self.score_matrix):
                data = {
                    "subject_id": f"model{i}",
                    "responses": {f"q{j+1}": int(score) for j, score in enumerate(row)}
                }
                f.write(json.dumps(data) + '\n')
        return score_path
        
    def read_attribute(self,path):
        with open(path,"r") as f:
            data = json.load(f)
        self.ability = data["ability"]
        self.diff = data["diff"]
        self.disc = data["disc"]
        return data