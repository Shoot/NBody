def paint(r1_sol, r2_sol, r3_sol, r4_sol):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    # Plot the orbits of the three bodies
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], color="mediumblue")
    ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="red")
    ax.plot(r3_sol[:, 0], r3_sol[:, 1], r3_sol[:, 2], color="gold")
    ax.plot(r4_sol[:, 0], r4_sol[:, 1], r4_sol[:, 2], color="silver")
    ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2], color="darkblue", marker="o", s=80, label="Body 1")
    ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2], color="darkred", marker="o", s=80, label="Body 2")
    ax.scatter(r3_sol[-1, 0], r3_sol[-1, 1], r3_sol[-1, 2], color="goldenrod", marker="o", s=80, label="Body 3")
    ax.scatter(r4_sol[-1, 0], r4_sol[-1, 1], r4_sol[-1, 2], color="cyan", marker="o", s=80, label="Body 4")
    ax.set_xlabel("x-coordinate", fontsize=14)
    ax.set_ylabel("y-coordinate", fontsize=14)
    ax.set_zlabel("z-coordinate", fontsize=14)
    ax.set_title("Visualization of orbits of stars in a 3-body system\n", fontsize=14)
    ax.legend(loc="upper left", fontsize=14)
    
    # Animate the orbits of the three bodies
    
    
    # Make the figure
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    
    # Create new arrays for animation, this gives you the flexibility
    # to reduce the number of points in the animation if it becomes slow
    # Currently set to select every 4th point
    r1_sol_anim = r1_sol[::1, :].copy()
    r2_sol_anim = r2_sol[::1, :].copy()
    r3_sol_anim = r3_sol[::1, :].copy()
    r4_sol_anim = r4_sol[::1, :].copy()

    # Set initial marker for planets, that is, blue,red and green circles at the initial positions
    head1 = [ax.scatter(r1_sol_anim[0, 0], r1_sol_anim[0, 1], r1_sol_anim[0, 2], color="darkblue", marker="o", s=80,
                        label="Body 1")]
    head2 = [ax.scatter(r2_sol_anim[0, 0], r2_sol_anim[0, 1], r2_sol_anim[0, 2], color="darkred", marker="o", s=80,
                        label="Body 2")]
    head3 = [ax.scatter(r3_sol_anim[0, 0], r3_sol_anim[0, 1], r3_sol_anim[0, 2], color="goldenrod", marker="o", s=80,
                        label="Body 3")]
    head4 = [ax.scatter(r4_sol_anim[0, 0], r4_sol_anim[0, 1], r4_sol_anim[0, 2], color="cyan", marker="o", s=80,
                        label="Body 4")]
    
    
    # Create a function Animate that changes plots every frame (here "i" is the frame number)
    def Animate(i, head1, head2, head3, head4):
        # Remove old markers
        head1[0].remove()
        head2[0].remove()
        head3[0].remove()
        head4[0].remove()

        # Plot the orbits (every iteration we plot from initial position to the current position)
        trace1 = ax.plot(r1_sol_anim[:i, 0], r1_sol_anim[:i, 1], r1_sol_anim[:i, 2], color="mediumblue")
        trace2 = ax.plot(r2_sol_anim[:i, 0], r2_sol_anim[:i, 1], r2_sol_anim[:i, 2], color="red")
        trace3 = ax.plot(r3_sol_anim[:i, 0], r3_sol_anim[:i, 1], r3_sol_anim[:i, 2], color="gold")
        trace4 = ax.plot(r4_sol_anim[:i, 0], r4_sol_anim[:i, 1], r4_sol_anim[:i, 2], color="silver")

        # Plot the current markers
        head1[0] = ax.scatter(r1_sol_anim[i - 1, 0], r1_sol_anim[i - 1, 1], r1_sol_anim[i - 1, 2], color="darkblue",
                              marker="o", s=100)
        head2[0] = ax.scatter(r2_sol_anim[i - 1, 0], r2_sol_anim[i - 1, 1], r2_sol_anim[i - 1, 2], color="darkred",
                              marker="o", s=100)
        head3[0] = ax.scatter(r3_sol_anim[i - 1, 0], r3_sol_anim[i - 1, 1], r3_sol_anim[i - 1, 2], color="goldenrod",
                              marker="o", s=100)
        head4[0] = ax.scatter(r4_sol_anim[i - 1, 0], r4_sol_anim[i - 1, 1], r4_sol_anim[i - 1, 2], color="cyan",
                              marker="o", s=100)
        return trace1, trace2, trace3, trace4, head1, head2, head3, head4,
    
    
    # Some beautifying
    ax.set_xlabel("x-coordinate", fontsize=14)
    ax.set_ylabel("y-coordinate", fontsize=14)
    ax.set_zlabel("z-coordinate", fontsize=14)
    ax.set_title("Visualization of orbits of stars in a 4-body system\n", fontsize=14)
    ax.legend(loc="upper left", fontsize=14)
    
    # If used in Jupyter Notebook, animation will not display only a static image will display with this command
    # anim_2b = animation.FuncAnimation(fig,Animate_2b,frames=1000,interval=5,repeat=False,blit=False,fargs=(h1,h2))
    
    
    # Use the FuncAnimation module to make the animation
    repeatanim = animation.FuncAnimation(fig, Animate, frames=300, interval=10, repeat=False, blit=False,
                                         fargs=(head1, head2, head3, head4))

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=4000)

    # To save animation to disk, enable this command
    repeatanim.save("FourBodyProblem.mp4", writer=writer)