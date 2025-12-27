import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import argparse
import os

@dataclass
class Arena:
    xmin: float = -5.0
    xmax: float = 55.0
    ymin: float = -20.0
    ymax: float = 25.0

@dataclass
class RectObstacle:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def xmin(self): return self.cx - self.w / 2
    @property
    def xmax(self): return self.cx + self.w / 2
    @property
    def ymin(self): return self.cy - self.h / 2
    @property
    def ymax(self): return self.cy + self.h / 2

@dataclass
class Params:
    dt: float = 0.1
    max_steps: int = 40000  

    speed: float = 1.2
    turn_sigma_forage: float = np.deg2rad(18)
    turn_sigma_home: float = np.deg2rad(6)

    dist_noise_rate: float = 0.03
    heading_noise_sigma: float = np.deg2rad(2)

    nest_radius: float = 1.0
    food_radius: float = 1.2
    food_min_dist_from_nest: float = 12.0

    food_sense_radius: float = 12.0
    food_attract_gain: float = 0.25

    obs_influence: float = 7.0
    obs_turn_gain: float = 0.65
    obs_margin: float = 0.25

    trail_max: int = 12000


def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def reflect_if_outside(xy, heading, arena: Arena):
    x, y = xy

    if x < arena.xmin:
        x = arena.xmin + (arena.xmin - x)
        heading = np.pi - heading
    elif x > arena.xmax:
        x = arena.xmax - (x - arena.xmax)
        heading = np.pi - heading

    if y < arena.ymin:
        y = arena.ymin + (arena.ymin - y)
        heading = -heading
    elif y > arena.ymax:
        y = arena.ymax - (y - arena.ymax)
        heading = -heading

    return np.array([x, y]), wrap_angle(heading)

def closest_point_on_rect(px, py, r: RectObstacle):
    cx = np.clip(px, r.xmin, r.xmax)
    cy = np.clip(py, r.ymin, r.ymax)
    return cx, cy

def inside_rect(px, py, r: RectObstacle):
    return (r.xmin <= px <= r.xmax) and (r.ymin <= py <= r.ymax)

def push_out_of_rect(px, py, r: RectObstacle, margin=0.1):
    dl = abs(px - r.xmin)
    dr = abs(r.xmax - px)
    db = abs(py - r.ymin)
    dt = abs(r.ymax - py)
    m = min(dl, dr, db, dt)

    if m == dl:
        px = r.xmin - margin
    elif m == dr:
        px = r.xmax + margin
    elif m == db:
        py = r.ymin - margin
    else:
        py = r.ymax + margin
    return px, py

def place_food(rng, arena: Arena, nest_xy, min_dist, obstacles):
    for _ in range(20000):
        fx = rng.uniform(arena.xmin + 2, arena.xmax - 2)
        fy = rng.uniform(arena.ymin + 2, arena.ymax - 2)

        if np.hypot(fx - nest_xy[0], fy - nest_xy[1]) < min_dist:
            continue

        ok = True
        for ob in obstacles:
            if inside_rect(fx, fy, ob):
                ok = False
                break
        if ok:
            return np.array([fx, fy])

    return np.array([arena.xmax - 5, 0.0])

def obstacle_avoid_turn(ant_xy, heading, obstacles, P: Params):
    x, y = ant_xy
    turn = 0.0

    for ob in obstacles:
        cx, cy = closest_point_on_rect(x, y, ob)
        vx, vy = x - cx, y - cy
        d = np.hypot(vx, vy)

        if inside_rect(x, y, ob):
            vx, vy = x - ob.cx, y - ob.cy
            d = max(np.hypot(vx, vy), 1e-6)

        d = max(d, 1e-6)
        influence = P.obs_influence

        if d <= influence:
            away_dir = np.arctan2(vy, vx)
            err = wrap_angle(away_dir - heading)
            strength = (influence - d) / influence
            turn += P.obs_turn_gain * strength * err

    return np.clip(turn, -np.deg2rad(25), np.deg2rad(25))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save_frames", type=str, default="", help="folder to save PNG frames")
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    arena = Arena()
    P = Params()

    
    obstacles = [
        RectObstacle(18,  8,  3.0, 16.0),
        RectObstacle(28, -6,  3.0, 18.0),
        RectObstacle(40, 10,  3.0, 16.0),
        RectObstacle(12, -10, 12.0, 3.0),
        RectObstacle(35,  0,  12.0, 3.0),
    ]

    
    nest = np.array([0.0, 0.0])
    food = place_food(rng, arena, nest, P.food_min_dist_from_nest, obstacles)

    ant_xy = nest.copy()
    heading = rng.uniform(-np.pi, np.pi)

    hv_est = np.array([0.0, 0.0])

    mode = "FORAGE"
    found_food = False

    forage_trail, home_trail = [], []

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for step in range(P.max_steps):
        dist_to_food = np.linalg.norm(ant_xy - food)
        dist_to_nest = np.linalg.norm(ant_xy - nest)

    
        if mode == "FORAGE":
            heading += rng.normal(0.0, P.turn_sigma_forage)

            
            if dist_to_food < P.food_sense_radius:
                desired = np.arctan2(food[1] - ant_xy[1], food[0] - ant_xy[0])
                heading += P.food_attract_gain * wrap_angle(desired - heading)

        else:  
            desired = np.arctan2(hv_est[1], hv_est[0])
            heading += 0.35 * wrap_angle(desired - heading) + rng.normal(0.0, P.turn_sigma_home)

    
        heading += obstacle_avoid_turn(ant_xy, heading, obstacles, P)
        heading = wrap_angle(heading)

    
        ds_true = P.speed * P.dt
        ant_xy = ant_xy + ds_true * np.array([np.cos(heading), np.sin(heading)])
        ant_xy, heading = reflect_if_outside(ant_xy, heading, arena)

    
        for ob in obstacles:
            if inside_rect(ant_xy[0], ant_xy[1], ob):
                px, py = push_out_of_rect(ant_xy[0], ant_xy[1], ob, margin=P.obs_margin)
                ant_xy = np.array([px, py])
                heading = wrap_angle(heading + np.pi/2 * rng.choice([-1, 1]))


        ds_meas = ds_true * (1.0 + rng.normal(0.0, P.dist_noise_rate))
        heading_meas = heading + rng.normal(0.0, P.heading_noise_sigma)
        hv_est = hv_est - ds_meas * np.array([np.cos(heading_meas), np.sin(heading_meas)])

        
        dist_to_food = np.linalg.norm(ant_xy - food)
        dist_to_nest = np.linalg.norm(ant_xy - nest)

        if mode == "FORAGE":
            forage_trail.append(ant_xy.copy())
            if dist_to_food <= P.food_radius:
                mode = "HOME"
                found_food = True
                home_trail.clear()
        else:
            home_trail.append(ant_xy.copy())
            if found_food and dist_to_nest <= P.nest_radius:
                pass

        
        if len(forage_trail) > P.trail_max:
            del forage_trail[:len(forage_trail) - P.trail_max]
        if len(home_trail) > P.trail_max:
            del home_trail[:len(home_trail) - P.trail_max]

        ax.cla()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(arena.xmin, arena.xmax)
        ax.set_ylim(arena.ymin, arena.ymax)
        ax.set_title("Desert Ant–Inspired Navigation: Path Integration with Obstacle Avoidance")

        for ob in obstacles:
            ax.add_patch(plt.Rectangle((ob.xmin, ob.ymin), ob.w, ob.h, alpha=0.35))

        if forage_trail:
            ft = np.array(forage_trail)
            ax.plot(ft[:, 0], ft[:, 1], "-", lw=1.5, label="forage trail")
        if home_trail:
            ht = np.array(home_trail)
            ax.plot(ht[:, 0], ht[:, 1], "-", lw=2.0, label="home trail")

        ax.plot(nest[0], nest[1], "ks", ms=8, label="nest")
        ax.plot(food[0], food[1], "*", ms=12, label="food")
        ax.plot(ant_xy[0], ant_xy[1], "o", ms=6, label="ant")

        hv_draw = hv_est.copy()
        hv_norm = np.linalg.norm(hv_draw)
        if hv_norm > 1e-6:
            max_len = 12.0
            if hv_norm > max_len:
                hv_draw = hv_draw * (max_len / hv_norm)
        else:
            hv_draw[:] = 0.0

        ax.quiver([ant_xy[0]], [ant_xy[1]], [hv_draw[0]], [hv_draw[1]],
                  angles="xy", scale_units="xy", scale=1.0, width=0.004)

        ax.text(
            0.02, 0.98,
            f"step: {step}\nmode: {mode}\nfound_food: {found_food}\n"
            f"|HV_est|: {np.linalg.norm(hv_est):.2f}\n"
            f"dist to nest: {dist_to_nest:.2f}\ndist to food: {dist_to_food:.2f}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", alpha=0.2)
        )

        ax.legend(loc="lower right")

        if args.save_frames:
            fig.savefig(os.path.join(args.save_frames, f"frame_{step:05d}.png"), dpi=120)

        if not args.no_show:
            plt.pause(0.001)

        if mode == "HOME" and found_food and dist_to_nest <= P.nest_radius:
            break

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
