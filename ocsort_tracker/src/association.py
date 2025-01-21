import os
import time

import numpy as np


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)                                              
    return(o)  


def giou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: ground truth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)  
    iou = wh / union

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1 
    hc = yyc2 - yyc1 
    assert((wc > 0).all() and (hc > 0).all())
    area_enclose = wc * hc 
    giou = iou - (area_enclose - union) / area_enclose
    giou = (giou + 1.)/2.0 # resize from (-1,1) to (0,1)
    return giou


def diou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 
    iou = wh / union
    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / outer_diag

    return (diou + 1) / 2.0 # resize from (-1,1) to (0,1)

def ciou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 
    iou = wh / union

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    # prevent dividing over zero. add one pixel shift
    h2 = h2 + 1.
    h1 = h1 + 1.
    arctan = np.arctan(w2/h2) - np.arctan(w1/h1)
    v = (4 / (np.pi ** 2)) * (arctan ** 2)
    S = 1 - iou 
    alpha = v / (S+v)
    ciou = iou - inner_diag / outer_diag - alpha * v
    
    return (ciou + 1) / 2.0 # resize from (-1,1) to (0,1)


def ct_dist(bboxes1, bboxes2):
    """
        Measure the center distance between two sets of bounding boxes,
        this is a coarse implementation, we don't recommend using it only
        for association, which can be unstable and sensitive to frame rate
        and object speed.
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    ct_dist = np.sqrt(ct_dist2)

    # The linear rescaling is a naive version and needs more study
    ct_dist = ct_dist / ct_dist.max()
    return ct_dist.max() - ct_dist # resize to (0,1)



def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:,0] + dets[:,2])/2.0, (dets[:,1]+dets[:,3])/2.0
    CX2, CY2 = (tracks[:,0] + tracks[:,2]) /2.0, (tracks[:,1]+tracks[:,3])/2.0
    dx = CX1 - CX2 
    dy = CY1 - CY2 
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm 
    dy = dy / norm
    return dy, dx # size: num_track x num_det


def euclidean_distance_batch(detections, trackers):
    # Calculate center points of detections
    det_centers = np.stack([(detections[:, 0] + detections[:, 2]) / 2,
                            (detections[:, 1] + detections[:, 3]) / 2], axis=1)
    # Calculate center points of trackers
    trk_centers = np.stack([(trackers[:, 0] + trackers[:, 2]) / 2,
                            (trackers[:, 1] + trackers[:, 3]) / 2], axis=1)
    # Compute pairwise Euclidean distances
    dists = np.linalg.norm(det_centers[:, None, :] - trk_centers[None, :, :], axis=2)
    return dists


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))




def associate(detections, trackers, class_id, iou_threshold, velocities, previous_obs, vdc_weight):
    # Only proceeds if there is active tracks
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    """
    This computes the difference in angle between:
        + The predicted velocity direction of the tracked objects (inertia_X, inertia_Y).
        + The direction vectors from the tracks to the detections (X, Y).
    """
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi
    """ Add 0.5 to allow more angular tolerance"""

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:,4]<0)] = 0

    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)

    # iou_matrix = iou_matrix * scores # a trick some items works, we don't encourage this

    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    # """
    #     With multiple categories, generate the cost for category mismatch
    # """
    # num_dets = detections.shape[0]
    # num_trk = trackers.shape[0]
    # cate_matrix = np.zeros((num_dets, num_trk))
    # for i in range(num_dets):
    #     for j in range(num_trk):
    #         if class_id[i] != trackers[j, 4]:
    #             cate_matrix[i][j] = -1e6
    #
    # cost_matrix = - iou_matrix - angle_diff_cost - cate_matrix

    """
    Penalise the cost matrix with the detection confidence of the last observed box of the active track.
    Track with goodness of iou and angle diff is high with low confidence is less favoured now.
    """
    # Step 1: Extract track confidence from previous observations
    track_confidence = previous_obs[:, -1]  # Extract confidence values (last column)
    track_confidence = np.maximum(track_confidence, 1e-6)  # Avoid division by zero

    # Step 2: Repeat confidence values to match the shape of the cost matrix
    confidence_weights = np.repeat(track_confidence[:, np.newaxis], detections.shape[0], axis=1).T

    # Step 3: Adjust the cost matrix
    adjusted_cost_matrix = -(iou_matrix + angle_diff_cost) * confidence_weights


    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(adjusted_cost_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    # Add the current detections to unmatched list if no existing trackers is matched with them in the current frame
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)

    # Add the existed trackers to unmatched list if no high-conf detections is matched with them in the current frame
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    # filter out matched pair with low IOU before confirm
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)































#
# def associate_kitti(detections, trackers, det_cates, iou_threshold,
#         velocities, previous_obs, vdc_weight):
#     if(len(trackers)==0):
#         return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
#
#     """
#         Cost from the velocity direction consistency
#     """
#     Y, X = speed_direction_batch(detections, previous_obs)
#     inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
#     inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
#     inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
#     diff_angle_cos = inertia_X * X + inertia_Y * Y
#     diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
#     diff_angle = np.arccos(diff_angle_cos)
#     diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi
#
#     valid_mask = np.ones(previous_obs.shape[0])
#     valid_mask[np.where(previous_obs[:,4]<0)]=0
#     valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
#
#     scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)
#     angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
#     angle_diff_cost = angle_diff_cost.T
#     angle_diff_cost = angle_diff_cost * scores
#
#     """
#         Cost from IoU
#     """
#     iou_matrix = iou_batch(detections, trackers)
#
#
#     """
#         With multiple categories, generate the cost for category mismatch
#     """
#     num_dets = detections.shape[0]
#     num_trk = trackers.shape[0]
#     cate_matrix = np.zeros((num_dets, num_trk))
#     for i in range(num_dets):
#             for j in range(num_trk):
#                 if det_cates[i] != trackers[j, 4]:
#                         cate_matrix[i][j] = -1e6
#
#     cost_matrix = - iou_matrix -angle_diff_cost - cate_matrix
#
#     if min(iou_matrix.shape) > 0:
#         a = (iou_matrix > iou_threshold).astype(np.int32)
#         if a.sum(1).max() == 1 and a.sum(0).max() == 1:
#             matched_indices = np.stack(np.where(a), axis=1)
#         else:
#             matched_indices = linear_assignment(cost_matrix)
#     else:
#         matched_indices = np.empty(shape=(0,2))
#
#     unmatched_detections = []
#     for d, det in enumerate(detections):
#         if(d not in matched_indices[:,0]):
#             unmatched_detections.append(d)
#     unmatched_trackers = []
#     for t, trk in enumerate(trackers):
#         if(t not in matched_indices[:,1]):
#             unmatched_trackers.append(t)
#
#     #filter out matched with low IOU
#     matches = []
#     for m in matched_indices:
#         if(iou_matrix[m[0], m[1]]<iou_threshold):
#             unmatched_detections.append(m[0])
#             unmatched_trackers.append(m[1])
#         else:
#             matches.append(m.reshape(1,2))
#     if(len(matches)==0):
#         matches = np.empty((0,2),dtype=int)
#     else:
#         matches = np.concatenate(matches,axis=0)
#
#     return matches, np.array(unmatched_detections), np.array(unmatched_trackers)











# def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
#     """
#     Assigns detections to tracked object (both represented as bounding boxes)
#     Returns 3 lists of matches, unmatched_detections and unmatched_trackers
#     """
#     if(len(trackers)==0):
#         return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
#
#     iou_matrix = iou_batch(detections, trackers)
#
#     if min(iou_matrix.shape) > 0:
#         a = (iou_matrix > iou_threshold).astype(np.int32)
#         if a.sum(1).max() == 1 and a.sum(0).max() == 1:
#             matched_indices = np.stack(np.where(a), axis=1)
#         else:
#             matched_indices = linear_assignment(-iou_matrix)
#     else:
#         matched_indices = np.empty(shape=(0,2))
#
#     unmatched_detections = []
#     for d, det in enumerate(detections):
#         if(d not in matched_indices[:,0]):
#             unmatched_detections.append(d)
#     unmatched_trackers = []
#     for t, trk in enumerate(trackers):
#         if(t not in matched_indices[:,1]):
#             unmatched_trackers.append(t)
#
#     #filter out matched with low IOU
#     matches = []
#     for m in matched_indices:
#         if(iou_matrix[m[0], m[1]]<iou_threshold):
#             unmatched_detections.append(m[0])
#             unmatched_trackers.append(m[1])
#         else:
#             matches.append(m.reshape(1,2))
#     if(len(matches)==0):
#         matches = np.empty((0,2),dtype=int)
#     else:
#         matches = np.concatenate(matches,axis=0)
#
#     return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
