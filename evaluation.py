import numpy as np

from sklearn.metrics import roc_curve
from scipy.spatial.distance import cdist
from tqdm import tqdm

def evaluate(session, __embeddings, __gamma, person_set, embeddings, gt_embeddings, required_fpr=10e-6, threshold=0.0):
    """Evaluate TPR and average positive distance
    Args:
       session: tensorflow session to calculate weights of each embedding inside track
       __embeddings: tf.placeholder to bind track feature embeddings
       __gamma: tf.Tensor - evaluate to obtain embedding weights
       embeddings: float32 numpy array (<train set image count>, <embedding_size>) - embeddings obtained by backbone model
       gt_embeddings: embeddings of ground-truth images obtained by production model
       threshold: skip embeddings with low gamma rate (gamma < threshold)
    """
    distances = []
    labels = []
    negative_pair_count = 0
    positive_pair_count = 0
    for person in tqdm(person_set, total=len(person_set)):
        #Calculate aggregated embedding for each person's track
        aggregated_embeddings = []
        for track in person.tracked_indicies:  
            """        
            aggregated_embedding = embeddings[track].mean(axis=0)    
            """
            #gamma can be calculated based on embeddings, obtained by large powerfull model        
            gamma = session.run(__gamma, feed_dict={ __embeddings: embeddings[track] })
            gamma *= (gamma >= threshold).astype(np.float32)
            gamma /= np.sum(gamma)

            #next gamma can be used to get weighted average of embeddings, obtained by production model
            weighted_track_embeddings = np.multiply(gamma, embeddings[track])

            assert(np.isfinite(weighted_track_embeddings).all())
            assert(np.sum(np.isnan(weighted_track_embeddings)) == 0)

            aggregated_embedding = np.sum(weighted_track_embeddings, axis=0)
            
            assert(np.isfinite(aggregated_embedding).all())
            assert(np.sum(np.isnan(aggregated_embedding)) == 0)
            
            curr_norm = np.linalg.norm(aggregated_embedding)
            if curr_norm > 0:
                aggregated_embedding = aggregated_embedding / curr_norm
            aggregated_embeddings.append(aggregated_embedding)
        aggregated_embeddings = np.stack(aggregated_embeddings)
        #Calculate distances from all ground-truth image embeddings to each track embedding  
        pairwise_distances = cdist(gt_embeddings, aggregated_embeddings)   
        assert(np.isfinite(pairwise_distances).all())   
        assert(np.sum(np.isnan(pairwise_distances)) == 0)  

        positive_distances = pairwise_distances[person.gt_indicies].reshape(-1)
        positive_pair_count += len(positive_distances)
        negative_distances = np.delete(pairwise_distances, person.gt_indicies, axis=0).reshape(-1) 
        negative_pair_count += len(negative_distances)

        assert(len(positive_distances) + len(negative_distances) == len(pairwise_distances.reshape(-1)))

        person_distances = np.hstack((positive_distances, negative_distances))
        person_labels = np.hstack((np.ones(len(positive_distances)), np.zeros(len(negative_distances))))

        distances.append(person_distances)
        labels.append(person_labels)

    distances = np.hstack(distances)
    labels = np.hstack(labels)

    assert(distances.shape == labels.shape)

    mean_positive_distance = np.mean(distances[labels == 1])

    fprs, tprs, thrs = roc_curve(labels, -1 * distances) 
 
    tprs_filtered = tprs[fprs <= required_fpr] 
    if len(tprs_filtered) == 0: 
        tpr = 0.0 
    else: 
        tpr = tprs_filtered[-1]         

    print('Negative pair count: ', negative_pair_count)
    print('Positive pair count: ', positive_pair_count)

    return tpr, mean_positive_distance
