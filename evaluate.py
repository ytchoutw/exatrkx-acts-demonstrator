#!/bin/python3

import argparse
from pathlib import Path

import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Exa.TrkX data generation/reconstruction script')
parser.add_argument('performance_file', help="track finding performance root file from ACTS", type=str)
args = vars(parser.parse_args())

resultFile = Path(args["performance_file"])

performance_particles = uproot.open("{}:track_finder_particles".format(resultFile))
performance_tracks = uproot.open("{}:track_finder_tracks".format(resultFile))

max_pT = 100 # GeV
min_pT = 0 # GeV
length_cut = 0

# Particle based metrics
particles_df = pd.DataFrame()
for key in performance_particles.keys():
    particles_df[key] = performance_particles[key].array(library="np")

particles_df["pT"] = np.hypot(particles_df.px, particles_df.py)
particles_df["eta"] = np.arctanh(particles_df.pz/np.hypot(particles_df.pT, particles_df.pz))
particles_df["theta"] = np.degrees(np.arctan(np.hypot(particles_df.pT, particles_df.pz)/particles_df.pz))
particles_df = particles_df[ particles_df["pT"] < max_pT ]
particles_df = particles_df[ particles_df["pT"] > min_pT ]
particles_df = particles_df[ particles_df["nhits"] > length_cut ]

# Track based metrics
tracks_df = pd.DataFrame()

for key in performance_tracks.keys():
    if key not in ['particle_id', 'particle_nhits_total', 'particle_nhits_on_track']:
        tracks_df[key] = performance_tracks[key].array(library="np")
    else:
        arrays = performance_tracks[key].array(library="np")
        tracks_df[key] = [ a[0] for a in arrays ]

tracks_df = tracks_df.rename(columns={
    "particle_id": "maj_particle_id",
    "particle_nhits_total": "maj_particle_nhits_total",
    "particle_nhits_on_track": "maj_particle_nhits_on_track",
})

tracks_df["purity"] = tracks_df["maj_particle_nhits_on_track"] / tracks_df["size"]
tracks_df["efficiency"] = tracks_df["maj_particle_nhits_on_track"] / tracks_df["maj_particle_nhits_total"]
tracks_df = tracks_df[ tracks_df["size"] > length_cut ]

# Map particle properties pT, eta & theta onto tracks
pT = np.hypot(particles_df["px"].to_numpy(), particles_df["py"].to_numpy())
p = np.hypot(pT, particles_df["pz"].to_numpy())
pL = particles_df["pz"].to_numpy()

tracks_df["pT"] = tracks_df["maj_particle_id"].map(dict(zip(particles_df["particle_id"], pT)))
tracks_df["eta"] = tracks_df["maj_particle_id"].map(dict(zip(particles_df["particle_id"], np.arctanh(pL/p))))
tracks_df["theta"] = tracks_df["maj_particle_id"].map(dict(zip(particles_df["particle_id"], np.degrees(np.arctan(p/pL)))))

# Efficiencies for particles
f = lambda x: x.maj_particle_nhits_on_track.max() / x.maj_particle_nhits_total[x.index[0]]
efficiency = tracks_df.groupby(["maj_particle_id", "event_id"]).apply(f).reset_index()
efficiency_dict = dict(zip(list(efficiency[["maj_particle_id", "event_id"]].itertuples(index=False, name=None)), efficiency[0]))
particles_df["efficiencies"] = particles_df.set_index(['particle_id', 'event_id']).index.map(efficiency_dict).fillna(0)

# Efficiency-thresholds for particles
for threshold in [0.5, 0.75, 0.9, 0.999]:
    reconstructed = tracks_df[ tracks_df["efficiency"] > threshold ][["maj_particle_id","event_id"]]

    particles_multi_index = particles_df.set_index(["particle_id","event_id"]).index
    reconstructed_multi_index = reconstructed.set_index(["maj_particle_id","event_id"]).index

    particles_df["reconstructed_{}".format(int(threshold*100))] = particles_multi_index.isin(reconstructed_multi_index).astype(int)

# Plot efficiency vs {eta, pT}
def plot_binned_2d(ax, x, y, bins, threshold=10, do_scatter=True, **plot_kwargs):
    hist, edges, _ = np.histogram2d(x, y, (bins,2))
    mask = np.sum(hist, axis=1) > threshold
    plotpoints = (edges[:-1] + np.diff(edges))[mask], hist[:,1][mask] / np.sum(hist, axis=1)[mask]
    line_plots = ax.plot(*plotpoints, **plot_kwargs)
    if do_scatter:
        ax.scatter(*plotpoints, color=line_plots[0]._color)

    return ax

fig, ax = plt.subplots(1,2)

ax[0] = plot_binned_2d(ax[0], particles_df.eta, particles_df.reconstructed_50, 10, label="50%")
ax[0] = plot_binned_2d(ax[0], particles_df.eta, particles_df.reconstructed_75, 10, label="75%")
ax[0] = plot_binned_2d(ax[0], particles_df.eta, particles_df.reconstructed_90, 10, label="90%")
ax[0] = plot_binned_2d(ax[0], particles_df.eta, particles_df.reconstructed_99, 10, label="100%")
ax[0].set_ylim(0,1)
ax[0].set_xlabel("$\eta$")
ax[0].set_ylabel("% reconstructed particles")

ax[1] = plot_binned_2d(ax[1], particles_df.pT, particles_df.reconstructed_50, 10, label="50%")
ax[1] = plot_binned_2d(ax[1], particles_df.pT, particles_df.reconstructed_75, 10, label="75%")
ax[1] = plot_binned_2d(ax[1], particles_df.pT, particles_df.reconstructed_90, 10, label="90%")
ax[1] = plot_binned_2d(ax[1], particles_df.pT, particles_df.reconstructed_99, 10, label="100%")
ax[1].set_ylim(0,1)
ax[1].set_xlabel("$p_T$")
ax[1].set_ylabel("% reconstructed particles")

fig.suptitle("Reconstructed particles with efficiency thresholds")
fig.tight_layout()

fig.savefig(resultFile.parent / "result.png")
#plt.show()
