# Track: `track_race`

The race track is fully self-contained in this package. Everything track-related
lives under `worlds/`:

```
worlds/
  my_track.world            # the Gazebo world; includes the track via model://track_race
  track_race/               # the track MODEL (travels with the package)
    model.config            # model name = track_race
    model.sdf               # references the meshes; uniform 0.001 mm->m scale, X/Y re-centre
    meshes/
      OMar.stl              # full mesh (collision)
      road.stl              # drivable surface (visual)
      surround.stl          # walls/border (visual)
      lines.stl             # lane lines (visual)
```

## How it resolves (why it works on any machine)

- `my_track.world` pulls the track in with `<include><uri>model://track_race</uri></include>`.
- `launch/mycar_autorace.launch.py` adds this package's `worlds/` dir to
  `GAZEBO_MODEL_PATH`, so `model://track_race` resolves to `worlds/track_race/`.
- Nothing uses an absolute path or the workspace folder name — it's all resolved
  from the ROS package name `auto_drive`. So the workspace can be named anything.

## Transfer to another ROS 2 Humble machine

Copy the whole `auto_drive` package into their `src/`, then:

```bash
colcon build --packages-select auto_drive
source install/setup.bash
ros2 launch auto_drive mycar_autorace.launch.py
```

No `~/.gazebo/models` step. The only requirement is that the package stays named
`auto_drive` (they already launch it by that name).

## Editing the track

Replace the STL(s) in `worlds/track_race/meshes/` (keep the same filenames, mm units),
then `colcon build` again. If a new mesh changes the track's center/size, update the
`<pose>` re-centre in `model.sdf` and the car spawn point in the launch file.
