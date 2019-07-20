import os
import pandas as pd

from pathlib import Path

class Person():
    """Stores embedding indicies for a given person"""
    def __init__(self, person_id):
        """
           Args:
               person_id: person id
        """
        self.person_id = person_id
        #indicies: list of integers, each integer is index of person's face embedding in the embeddings array
        self.indicies = [] 
        #tracked_indicies: list of lists of integers, each list contains indicies of person's face embedding in a single track
        self.tracked_indicies = []
        #gt_indicies: list of integers, each integer is index of person's ground-truth face embedding in the ground-truth embeddings array    
        self.gt_indicies = [] 

    def __len__(self):
        return len(self.embedding_indicies)    


def read_dataset_from_file(train_file_name, gt_file_name=None):  
    """
        Read list of `Person` from specified csv file. File should contain columns named 'person_id' and 'track_id'
        Suppose, that zero-based line number equals to index of corresponding face embedding in the embeddings array
        Colums should be separated by ',' character. 
        Ground-truth set is optional - csv file should contain column named 'person_id'

        Args:
            train_file_name: path to csv file, string
            gt_file_name: path to csv file, string
        Returns:
            dataset: list of `Person`
    """
    dataset = []

    train_data_frame = pd.read_csv(train_file_name)
    person_ids = train_data_frame.person_id.values
    track_ids = train_data_frame.track_id.values

    #temp dictionary - keys are person ids, values are `Person` 
    person_dict = dict()
    #keys are track ids, values are lists of integers
    track_dict = dict()
    #keys are track ids, values are person ids
    track_to_person_dict = dict()
    for i, (person_id, track_id) in enumerate(zip(person_ids, track_ids)):
       
        if person_id not in person_dict:
            person = Person(person_id)
            person_dict[person_id] = person
            dataset.append(person)        

        person_dict[person_id].indicies.append(i)

        if track_id not in track_dict:
            track_dict[track_id] = []
        track_dict[track_id] += [i]

        if track_id not in track_to_person_dict:
            track_to_person_dict[track_id] = person_id

    for track_id, person_id in track_to_person_dict.items():
        person_dict[person_id].tracked_indicies.append(track_dict[track_id])

    if gt_file_name is not None:
        gt_data_frame = pd.read_csv(gt_file_name)
        person_ids = gt_data_frame.person_id.values
        for i, person_id in enumerate(person_ids):
            person_dict[person_id].gt_indicies.append(i)

    return dataset
