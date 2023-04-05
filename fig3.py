from matplotlib import pyplot as plt
from pycirclize import Circos

from dagip.nipt.binning import ChromosomeBounds


# Initialize circos sectors
bounds = ChromosomeBounds.get_1mb()
sectors = {f'chr {c + 1}': bounds[c + 1] - bounds[c] for c in range(22)}
circos = Circos(sectors, space=4)

for c, sector in enumerate(circos.sectors):

    # Plot sector axis & name text
    sector.axis(fc="none", ls="dashdot", lw=2, ec="black", alpha=0.5)
    sector.text(sector.name, size=9)

    # Set Track01 (Radius: 75 - 100)
    track1 = sector.add_track((75, 100))
    track1.axis(fc="tomato", alpha=0.5)

    # Set Track02 (Radius: 45 - 70)
    track2 = sector.add_track((45, 70))
    track2.axis(fc="cyan", alpha=0.5)

    # Set Track03 (Radius: 15 - 40)
    track3 = sector.add_track((15, 40))
    track3.axis(fc="lime", alpha=0.5)


fig = circos.plotfig()
plt.show()