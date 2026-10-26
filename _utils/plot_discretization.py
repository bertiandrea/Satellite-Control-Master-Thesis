import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation


def generate_orientations(num_envs=4096):
    n = round(np.cbrt(num_envs))
    if num_envs <= 0 or n**3 != num_envs:
        raise ValueError(f"num_envs must be a perfect cube, got {num_envs}")

    # Fibonacci sphere
    i = np.arange(n**2)
    x = np.linspace(1, -1, n**2)
    r = np.sqrt(1 - x**2)
    theta = np.pi * (1 + np.sqrt(5)) * i

    # Euler angles (ZYX)
    yaw = np.arctan2(r * np.cos(theta), x)
    pitch = np.arcsin(-r * np.sin(theta))
    roll = np.linspace(0, 2 * np.pi, n, endpoint=False)

    angles = np.column_stack((
        np.repeat(yaw, n),
        np.repeat(pitch, n),
        np.tile(roll, n**2)
    ))

    return Rotation.from_euler("ZYX", angles), angles


# Generate orientations
orientations, angles = generate_orientations()
points = orientations.apply([1, 0, 0])
angles_deg = np.degrees(angles)

hover = (
    "Yaw: %{customdata[0]:.2f}°<br>"
    "Pitch: %{customdata[1]:.2f}°"
    "<extra></extra>"
)

# Plot orientations and goal
fig = go.Figure([
    go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=5, color="black"),
        customdata=angles_deg,
        hovertemplate=hover,
        name="Orientations"
    ),
    go.Scatter3d(
        x=points[:1, 0], y=points[:1, 1], z=points[:1, 2],
        mode="markers",
        marker=dict(size=10, color="red"),
        customdata=angles_deg[:1],
        hovertemplate="Goal<br>" + hover,
        name="Goal"
    )
])

fig.update_layout(
    template="plotly_white",
    legend=dict(
        font=dict(size=50),
        itemsizing="constant"
    ),
    scene=dict(
        bgcolor="white",
        xaxis=dict(title="X", showticklabels=False),
        yaxis=dict(title="Y", showticklabels=False),
        zaxis=dict(title="Z", showticklabels=False)
    )
)

fig.write_html("orientations_3d.html", auto_open=True)