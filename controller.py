"""Baseline: a hand-written trot with steering. No learning, about 60 lines of maths.

Gait: diagonal pairs (back-right + front-left, back-left + front-right) take turns. Over one cycle each foot
spends half the time in the air, swinging forward and lifting, and half on the ground, sweeping back, which
pushes the body forward. Each foot's target offset from the standing pose (forward dx, up dz) becomes joint angles
with the leg's approximate Jacobian near the standing pose: +1 rad of hip moves the foot ~0.86 m forward, +1 rad
of knee moves it ~0.37 m forward and ~0.25 m up.

Steering: a heading controller turns back towards the track's centreline (y = 0) by lengthening the stride on
one side and shortening it on the other. Terrain: the foot lift grows with the highest terrain in the height scan
just ahead, so it steps over hurdles and up stairs.

Balance: swing feet land further along the way the body is drifting (Raibert's foot-placement rule), and the
legs on the high side shorten to keep the body level. The numbers come from a search that kept it walking when
any one of them changes by 10-20%, so you can tune them by hand; bigger or combined changes can still topple it.

It is deliberately simple: it walks about 0.6 m/s on flat ground and has no foothold planning, so the
obstacles stop it. Beat it.
"""
import numpy as np

from sprout_challenge import Controller

HIP_DX, KNEE_DX, KNEE_DZ = 0.86, 0.37, 0.25      # foot motion [m] per rad, near the standing pose
ABD_DY = 0.83                                    # m sideways (outward) per rad of abduction
PHASE = np.array([0.0, 0.5, 0.0, 0.5])           # BR, BL, FL, FR: diagonals in step
LEFT = np.array([False, True, True, False])


class Trot(Controller):
    freq = 2.35         # Hz, steps per second per leg
    stride = 0.1        # m, half the foot's sweep
    lift = 0.205        # m, foot lift on flat ground
    k_yaw, lean = 1.15, 0.46
    k_place = 0.76      # s: foot placement per m/s of velocity error (Raibert's rule)
    k_level = 0.37      # m of leg-length change per radian of body tilt

    def reset(self, info):
        self.pose = np.array(info["default_pose"], float)
        self.dt = info["dt"]
        self.scan_x = info["scan_points"][:, 0, 0]
        self.phase = 0.0

    def act(self, obs):
        self.phase = (self.phase + self.freq * self.dt) % 1.0
        # steering: turn towards a heading that leans back to the centreline
        want = np.clip(-self.lean * obs.position[1], -0.3, 0.3)
        turn = np.clip(self.k_yaw * (want - obs.yaw), -0.5, 0.5)            # > 0: turn left
        stride = self.stride * np.where(LEFT, 1 - turn, 1 + turn)
        # terrain ahead: the highest point 0.2 .. 1.0 m in front, relative to the ground under the body
        scan = obs.height_scan
        under = np.median(scan[(self.scan_x > -0.4) & (self.scan_x < 0.4)])
        ahead = scan[(self.scan_x > 0.2) & (self.scan_x < 1.0)].max()
        lift = self.lift + np.clip(ahead - under, 0.0, 0.25)

        u = (self.phase + PHASE) % 1.0
        swing = u < 0.5
        s = np.where(swing, u / 0.5, (u - 0.5) / 0.5)                       # 0 .. 1 within each half
        dx = np.where(swing, -stride + 2 * stride * (0.5 - 0.5 * np.cos(np.pi * s)), stride - 2 * stride * s)
        dz = np.where(swing, lift * np.sin(np.pi * s), 0.0)
        # balance: swing feet land further along the way the body is drifting, which catches the drift
        v_want = 4 * self.stride * self.freq
        land = np.where(swing, 0.5 - 0.5 * np.cos(np.pi * s), 0.0)       # 0 -> 1 over the swing
        dx = dx + land * self.k_place * (obs.base_lin_vel[0] - v_want)
        dy = land * self.k_place * obs.base_lin_vel[1]                    # +: foot to the left
        # posture: shorten the legs on the high side (raise those feet) to keep the body level
        nose_up, left_up = np.arcsin(np.clip(-obs.gravity[:2], -1, 1))
        front = np.array([-1.0, -1.0, 1.0, 1.0])
        side = np.where(LEFT, 1.0, -1.0)
        dz = dz + self.k_level * 0.33 * (front * nose_up + side * left_up)
        knee = dz / KNEE_DZ
        hip = (dx - KNEE_DX * knee) / HIP_DX
        abd = np.where(LEFT, dy, -dy) / ABD_DY                            # + abduction moves a foot outward
        q = self.pose.copy().reshape(4, 3)
        q[:, 0] += abd
        q[:, 1] += hip
        q[:, 2] += knee
        return q.ravel()
