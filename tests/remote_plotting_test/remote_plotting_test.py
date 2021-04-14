# Test remote plotting via X11Forwarding
# https://stackoverflow.com/questions/3453188/matplotlib-display-plot-on-a-remote-machine
# (answer from ValAyal)
# https://marketplace.visualstudio.com/items?itemName=spadin.remote-x11

# Other resources:
# https://adoni.github.io/2019/01/08/plot-on-server/
# https://superuser.com/questions/806637/xauth-not-creating-xauthority-file

# This was the solution: Remote X11 (as Remote SSH doesnt support X11 yet...)
# https://marketplace.visualstudio.com/items?itemName=spadin.remote-x11

from termcolor import colored, cprint

import matplotlib
matplotlib.use('tkagg')  #I had to use GTKAgg for this to work, GTK threw errors
import matplotlib.pyplot as plt

cprint("Plotting...", "yellow")

plt.plot([1, 2, 3])
plt.show()

cprint("Done!", "green")