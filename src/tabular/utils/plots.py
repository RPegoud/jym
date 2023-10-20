import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def animated_heatmap(
    q_values: np.ndarray,
    dims: list,
    agent_name: str = None,
    sample_freq=500,
    log_scale=False,
):
    # Compute the maximal Q-values along the action dimension
    max_q_values = q_values.max(axis=-1)
    if log_scale:
        sampled_q_values = np.log(max_q_values[::sample_freq])
    else:
        sampled_q_values = max_q_values[::sample_freq]  # Sample every 500 steps

    sampled_q_values = np.asarray(sampled_q_values)

    fig = go.Figure(
        data=[go.Heatmap(z=sampled_q_values[0])],
        layout=go.Layout(
            title=f"{agent_name}: Maximal Q-values for Step 0",
            title_x=0.5,
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="Play", method="animate", args=[None]),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
        ),
        frames=[
            go.Frame(
                data=[go.Heatmap(z=sampled_q_values[i])],
                layout=go.Layout(
                    title_text=f"{agent_name}: Maximal Q-values for Step {i*sample_freq}"
                ),
                traces=[0],
                name=f"frame_{i}",
            )
            for i in range(len(sampled_q_values))
        ],
    )

    fig.update_layout(
        title=f"{agent_name}: Maximal Q-values for Step 0",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        height=600,
        width=800,
        xaxis=dict(range=[0, dims[1]]),  # Set the range for x-axis based on grid_y
        yaxis=dict(range=[dims[0], 0], autorange=False),
    )

    # Decrease the duration for faster transitions (in milliseconds)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="relayout",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}}],
                    ),
                ],
            )
        ]
    )

    fig.show()


def plot_path(obs):
    df_obs = pd.DataFrame(obs)
    fig = px.imshow((pd.crosstab(df_obs[0], df_obs[1])), title="State visit count")
    fig.show()
