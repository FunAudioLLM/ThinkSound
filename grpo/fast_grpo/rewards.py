import io
import numpy as np
import torch
from collections import defaultdict



def LAION_clap_score(device):
    from .rewards_LAION_clap.py import CLAPRewardModel

    scorer = CLAPRewardModel(device=device)

    def _fn(audios, prompts, metadata):
        scores=scorer(audios,prompts)
        # change tensor to list
        return scores, {}

    return _fn



def meta_reward(device):
    from .reward_meta import AesPredictor

    scorer = AesPredictor(device=device)

    def _fn(audios, prompts, metadata):
        scores=scorer(audios)
        # change tensor to list
        return scores, {}

    return _fn


def meta_reward_CE(device):
    from .reward_meta import AesPredictor

    scorer = AesPredictor(device=device,eval_type="CE")

    def _fn(audios, prompts, metadata):
        scores=scorer(audios)
        # change tensor to list
        return scores, {}

    return _fn

def synch_reward(device):
    from .reward_synch import synchpredict
    scorer = synchpredict(device=device)

    def _fn(audios, prompts, metadata):
        scores=scorer(audios,metadata)
        # change tensor to list
        return scores, {}
    return _fn

def itd_reward(device):
    from .reward_itd import itdpredict
    scorer = itdpredict(device=device)

    def _fn(audios, prompts, metadata):
        scores=scorer(audios,metadata)
        # change tensor to list
        return scores, {}
    return _fn

def ms_clap(device):
    from .reward_ms_clap import MSCLAPRewardModel
    scorer = MSCLAPRewardModel(device=device)

    def _fn(audios, prompts, metadata):
        scores=scorer(audios,prompts)
        # change tensor to list
        return scores, {}
    return _fn
    


def multi_score(device, score_dict):


    score_functions = {
        "meta_reward_CE":meta_reward_CE,
        "ms_clap":ms_clap,
        "itd_reward":itd_reward,
        "synch_reward":synch_reward,
        'meta_reward':meta_reward,
        'LAION_clap': LAION_clap_score,
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

            
    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(audios, prompts, metadata, ref_audios=None, only_strict=True):
        total_scores = []
        score_details = {}
        #print(metadata)
        
        for score_name, weight in score_dict.items():
            scores, rewards = score_fns[score_name](audios, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        return score_details, {}

    return _fn

