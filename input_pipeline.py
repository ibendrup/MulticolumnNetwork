import itertools

from tqdm import tqdm 
from random import shuffle

import numpy as np
import tensorflow as tf

class Combination():
    """Stores the some indicies of embeddings for a given person"""
    def __init__(self, person_id, indicies):
        """
           Args:
               person_id: person id
               indicies: list of integers, each integer is index of person's face embedding in the embeddings array,
                         embeddings forms the combination
        """
        self.person_id = person_id        
        self.indicies = indicies      

def _data_generator(dataset, embeddings, combinations_per_person=1, combination_size=3):
    """This function will generate random embedding combinations and corresponding labels to compose train data for one epoch 
    Args:
        dataset: list of `Person` - whole training dataset, i-th element corresponds to person i
        embeddings: 2D array of person embeddings, each row is embedding
        combinations_per_person: int - required number of unique embedding combinations from each person
        combination_size: int - embedding combination size
    Returns:
        infinite generator with the following structure:
            combinations: list of integers - embedding indicies with length equal to len(dataset) x combinations_per_person x combination_size
            labels: list of integers - combination labels, length equal to len(dataset) x combinations_per_person
    """

    person_indicies = np.arange(len(dataset))
    while True:
        shuffle(person_indicies)

        combination_objects = []

        for person_id in person_indicies:
            person = dataset[person_id]
            
            for track in person.tracked_indicies:
                person_combinations = np.random.choice(track, (combinations_per_person, combination_size), replace=False)
                #Append combinations to list
                for person_combination in person_combinations:
                    combination_object = Combination(person_id, person_combination)
                    combination_objects.append(combination_object)
            """
            #Select `combinations_per_person` unique combinations for this person
            person_combinations = np.random.choice(person.indicies, (combinations_per_person, combination_size), replace=False)
            #Append combinations to list
            for person_combination in person_combinations:
                combination_object = Combination(person_id, person_combination)
                combination_objects.append(combination_object)
            """

        #Shuffle combination list to violate person order
        shuffle(combination_objects)                      

        for combination in combination_objects:
            yield embeddings[combination.indicies], combination.person_id


def __create_pipelnine(dataset, 
                       embeddings,
                       combinations_per_person, 
                       combination_size,
                       combinations_per_batch,
                       buffer_size=1):
    """Create infinite dataset of random images and labels to select embedding combinations
    Args:
        dataset: list of `Person` - whole training dataset, i-th element corresponds to person i
        embeddings: 2D array of person embeddings, each row is embedding
        combinations_per_person: int - required number of unique embedding combinations from each person
        combination_size: int - embedding combination size
        combinations_per_batch: int - required number of combinations in the batch
        buffer_size - maximum size of the prefetched image packages
    Returns:        
        __embeddings: float32 tf.Tensor of shape (combinations_per_batch x combination_size, <embedding size>) - train combinations
        __labels: tf.int32 Tensor of shape (combinations_per_batch) - combination labels
    """   

    _generator = lambda : _data_generator(dataset, embeddings,
                                          combinations_per_person, 
                                          combination_size)

    __dataset = tf.data.Dataset.from_generator(_generator, output_types=(tf.float32, tf.int32))
    __dataset = __dataset.shuffle(1000)
    __dataset = __dataset.prefetch(combinations_per_batch * buffer_size * 1000)
    __dataset = __dataset.batch(combinations_per_batch)
    __dataset = __dataset.prefetch(buffer_size)
    __dataset = __dataset.map(lambda e, l: (tf.reshape(e, (-1, embeddings.shape[1])), l))

    iterator = __dataset.make_one_shot_iterator()

    __next_op = iterator.get_next()

    __embeddings = __next_op[0]
    __labels = __next_op[1]

    return __embeddings, __labels        
