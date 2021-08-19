import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='GazeboSocialNav-v1',
    entry_point='gym_social.envs:SocialNavEnv1',
    reward_threshold=1.0,
    nondeterministic = True,
)
