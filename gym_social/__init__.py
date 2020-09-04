import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Custon-v0',
    entry_point='gym_social.envs:CustomEnv',
    reward_threshold=1.0,
    nondeterministic = True,
)

register(
    id='GazeboSocialNav-v0',
    entry_point='gym_social.envs:SocialNavEnv',
    reward_threshold=1.0,
    nondeterministic = True,
)
