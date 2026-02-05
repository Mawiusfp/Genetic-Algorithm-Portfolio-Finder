import os
import random
import sys
import threading
import sqlite3
import sys
from multiprocessing import Pool, Manager, Lock
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class IndividuoProyecto:
    def __init__(self, projects):
        self.projects = projects
        self.fitness = self.calculate_fitness()
        self.cost = sum(p.cost for p in self.projects)
 
    def __str__(self):
        names = ", ".join(p.name for p in self.projects)
        return f"Fitness: {self.fitness} | Cost: {self.cost} | Projects: {names} "
 
    def calculate_fitness(self):
        """Return fitness of project list"""
        return sum(p.profit for p in self.projects)
 
    def update_fitness(self):
        """Recalculate fitness of project list"""
        self.fitness = self.calculate_fitness()
        self.cost = sum(p.cost for p in self.projects)
 
    def mutate(self, projects, budget, probability, strength):
        """
            Apply a mutation to the project list, given a probability and a strength
            Remove n projects, the higher the strength the more projects to remove
        """
        # Check for empty project list
        # Roll for mutation probability
        if not self.projects or random.random() >= probability:
            return
        
        # Calculate amount of projects to remove
        amount_to_remove = max(1, int(len(self.projects) * strength))

        # Remove amount of projects calculated
        for _ in range(amount_to_remove):

            # Safeguard
            if not self.projects: break

            # Remove a random project in the list
            del self.projects[random.randint(0, len(self.projects) - 1)]

        # Update fitness with the shrinked project list
        self.update_fitness()

        # Create a copy of the projects list
        projects_shuf = projects[:]

        # Shuffle order of cloned list
        random.shuffle(projects_shuf)

        # Iterate over the shuffled project list
        for p in projects_shuf:

            # Check if project list + the current iterated project fits within the budget
            if p not in self.projects and ( self.cost + p.cost <= budget ):

                # Append the project and update proj list fitness
                self.projects.append(p)
                self.update_fitness()

class Proyecto:
    def __init__(self, name, cost, profit):
        self.name = name
        self.cost = float(cost.strip())
        self.profit = float(profit.strip())
 
    @staticmethod
    def raw_data_to_list(data):
        projects = []
        for d in data:
            if ',' in d:
                splits = d.split(',')
                projects.append(Proyecto(splits[0], splits[1], splits[2]))
        return projects



def get_random_individuo_proyecto(projects, budget):
    """Generate a random combination of projects that fit within the budget"""

    # Create a copy of projects
    available = projects[:]

    # Shuffle the copy of projects
    random.shuffle(available)

    # Keep track of the chosen projects and total cost
    selected, cost = [], 0

    # Iterate over all of the projects
    for p in available:

        # Check if adding the project to the list exceeds the max budget
        if cost + p.cost <= budget:

            # Add it to the list and update total cost
            selected.append(p)
            cost += p.cost

    return IndividuoProyecto(selected)
 
def remove_worst_and_replace(poblation, projects, budget, amount):
    """Replace a set amount of projects with a random combination of projects"""
    
    # Sort poblation by fitness asc
    poblation.sort(key=lambda ind: ind.fitness) 
    
    for _ in range(amount):
        poblation.pop(0) # Remove worst (index 0)
        poblation.append(get_random_individuo_proyecto(projects, budget)) # Append a new prj



def find_best_combination(projects, budget, epochs, poblation_size, natural_deselection, probability, strength, manager_goat, manager_history, lock):
    """
    Find the best combination of projects to fit within a budget.
    Generate a set of random combinations, get the best copy and try to improve it.
    From there get the weakest combinations and replace with complete random combinations to 
        not focus only on the first apparent best combination we find.
    """

    # Start by generating a randomized poblation
    poblation = [get_random_individuo_proyecto(projects, budget) for _ in range(poblation_size)]

    # Choose from the population the list of projects and mark it as the best to begin with
    local_best = IndividuoProyecto(list(poblation[0].projects))

    for epoch in range(epochs):

        # Get the highest fitness value inside the poblation
        iteration_best = max(poblation, key=lambda individuo: individuo.fitness)

        # Check if in this iteration there's a new best
        if iteration_best.fitness > local_best.fitness:

            # Set the newly found combination as the local best
            local_best = IndividuoProyecto(list(iteration_best.projects))

            # Check if the local best beats the GLOBAL goat
            if local_best.fitness > manager_goat['fitness']:
                with lock:
                    if local_best.fitness > manager_goat['fitness']:
                        # Update global goat
                        manager_goat.update({
                            'fitness': local_best.fitness,
                            'epoch': epoch,
                            'probability': probability,
                            'strength': strength,
                            'projects': str(local_best)
                        })
                        
                        # Add to history for the graph
                        manager_history.append({
                            'epoch': epoch,
                            'fitness': local_best.fitness,
                            'p': probability, 
                            's': strength
                        })

                        print(f"> NEW GOAT: {local_best.fitness} (P:{probability} S:{strength})")

        # Apply mutation
        for ind in poblation: 
            ind.mutate(projects, budget, probability, strength)

        # Remove the worse combinations
        remove_worst_and_replace(poblation, projects, budget, natural_deselection)

        # Sort the poblation by its fitness
        poblation.sort(key=lambda x: x.fitness)

        # Set first slot as the absolute best so we dont mutate it too soon
        poblation[0] = IndividuoProyecto(list(local_best.projects))


    return {
        'p': probability, 
        's': strength, 
        'fitness': local_best.fitness
    }


def create_graph(goat_history, final_results):
    # Convert shared list/dicts to DataFrames
    history_df = pd.DataFrame(list(goat_history))
    results_df = pd.DataFrame(final_results)

    fig = plt.figure(figsize=(18, 6))

    # --- 1. STEP PLOT (History) ---
    plt.subplot(1, 3, 1)
    if not history_df.empty:
        history_df = history_df.sort_values('fitness') # Sort to make the step look correct
        # Create a synthetic index 0..N for the step steps
        plt.step(range(len(history_df)), history_df['fitness'], where='post', color='green', linewidth=2)
        plt.title(f"Convergence History ({len(history_df)} breaks)")
        plt.xlabel("Record Broken Event #")
        plt.ylabel("Fitness Value")
    else:
        plt.text(0.5, 0.5, "No History Data", ha='center')
    plt.grid(True, alpha=0.3)

    # --- 2. HEATMAP ---
    plt.subplot(1, 3, 2)
    pivot_table = results_df.pivot(index='s', columns='p', values='fitness')
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Fitness Landscape (Strength vs Probability)")
    plt.xlabel("Probability")
    plt.ylabel("Strength")

    # --- 3. 3D PLOT ---
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title("Optimization Surface")
    ax.set_xlabel('Prob')
    ax.set_ylabel('Strength')
    ax.set_zlabel('Fitness')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()

def spawn_processes(shared_goat, shared_history, shared_lock, data, budget, epochs, poblation, natural_deselection, min, max):
    """Create separate processes working together to find the best combination of projects"""

    try:

        # Convert raw data to a list of proyectos
        projects = Proyecto.raw_data_to_list(data)

        # Store tasks to run
        tasks = []

        # Scale since iterating with decimals is not allowed in py
        # LEAVE AS IT IS, NO TOUCH
        _scale = 10
        
        # Create Tasks 
        for i in range(int(min * _scale), int(max * _scale) + 1):  # P
            for j in range(int(min * _scale), int(max * _scale) + 1):  # S

                tasks.append((
                    projects, budget, epochs, poblation, natural_deselection, 
                    i / _scale, j / _scale, 
                    shared_goat, shared_history, shared_lock
                ))
        
        # Randomize the order of tasks, (doesn't really matter, higher values always come up with the better results, 
        # if it works dont tryna fix it)
        random.shuffle(tasks)

        print(f"--- Running {len(tasks)} Simulations in Parallel ---")

        with Pool() as pool:
            # STARMAP executes the function using the tuple arguments
            final_results = pool.starmap(find_best_combination, tasks)

        
        # --- Final Report ---
        c = 60
        print("\n" + "="*c)
        print(f"Global Best Fitness: {shared_goat['fitness']}")
        print(f"Found with Params:   Prob {shared_goat['probability']} / Str {shared_goat['strength']}")
        print(f"Project Selection:   {shared_goat['projects']}")
        print("="*c)

        try:

            # Get all of the project names
            raw_projects_str = shared_goat['projects'].split("Projects: ")[1]

            # Split the projects and only get the name
            project_names = [p.strip() for p in raw_projects_str.split(",")]
            
            global projs_dict
            # Check if we should update the GUI
            
            if gui_active and listbox is not None:
                # Clear listbox and append every name
                listbox.delete(0, tk.END)
                for name in project_names:
                    if name in projs_dict:
                        cost, profit = projs_dict[name]
                        listbox.insert(
                            tk.END,
                            f"{name} | Cost: {cost} | Profit: {profit}"
                        )
                    else:
                        listbox.insert(tk.END, f"{name}")

            
            # Always try to create graphs, regardless of GUI mode
            try:
                create_graph(shared_history, final_results)

            except Exception as e:
                print(f"Graphing skipped or failed: {e}")
                
        except IndexError as e:
            print("Couldn't find permutations.")
            print("> Is there enough budget ?")
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error in main: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"ERROR: {e}")

def gui(shared_goat, shared_history, shared_lock):
    """Return a gui for the portfolio genetic algorythm"""
    global gui_active
    gui_active = True
    window = tk.Tk()
    window.title("GA Optimization") # Set title
    window.geometry("500x900") 
    window.resizable(False, False)
    window.config(background="#0A0A0A") # Background color

    def start():
        
        # Clear data
        shared_history[:] = []
        shared_goat.update({
            'fitness': -1, 
            'epoch': -1, 
            'probability': -1, 
            'strength': -1, 
            'projects': ""
        })


        try:
            # Get widget values
            f_path = entry_path.get()
            f_budget = float(entry_budget.get())
            f_min = float(min_value.get())
            f_max = float(max_value.get())
            f_epochs = int(entry_epochs.get())
            f_population = int(entry_poblation.get())
            f_desel = int(entry_deselection.get())

            global data, projs_dict
            # Read file
            with open(f_path, 'r', encoding='UTF-8') as f:
                data = f.read().split('\n')
                # remove empty lines
                data = [line for line in data if line.strip()]
                
            projs_dict = {}
                
            for line in data:
                temp = line.split(',')
                if len(temp) >= 3:  
                    project_name = temp[0]
                    value1 = temp[1]
                    value2 = temp[2]
                    projs_dict[project_name] = [value1, value2]

            # btn.config(state="disabled", text="Reading file...")
            # btn.config(state="disabled", text="RUNNING...")

            # Create a separate thread to prevent ui locking
            threading.Thread(
                target=spawn_processes,
                args=(
                    shared_goat, 
                    shared_history, 
                    shared_lock, 
                    data,
                    f_budget,
                    f_epochs,
                    f_population,
                    f_desel,
                    f_min,
                    f_max
                ), daemon=True
                ).start()
            
            """
            spawn_processes(
                shared_goat=shared_goat, 
                shared_history=shared_history, 
                shared_lock=shared_lock, 
                data=data,
                budget=f_budget,
                epochs=f_epochs,
                poblation=f_population,
                natural_deselection=f_desel,
                min=f_min,
                max=f_max
            )
            """

        except ValueError:
            messagebox.showerror("Error", "Check your inputs!")

    # UI setup
    ttk.Label(window, text="Algoritmo Genetico", font=("monospace", 24, "bold"), background="#0A0A0A", foreground="white").pack(pady=20)
    style = ttk.Style()

    style.configure("TScale",  
                troughcolor="#3c3c3c",
                sliderlength=20,
                orient=tk.HORIZONTAL,
                background="#0A0A0A",
                foreground="white",
                )

    """
    ====================================================================
        OLD SLIDERS, LEAVE AS IT IS JUST IN CASE    
        
        # Update sliders labels to show (10% - 20% ... 90% - 100%)
        def update_slider1(v):
            percent = round(float(v) * 100 / 10) * 10
            # label1.config(text=f"{percent}%")

        def update_slider2(v):
            percent = round(float(v) * 100 / 10) * 10
            # label2.config(text=f"{percent}%")

        # slider1 = ttk.Scale(window, from_=0.1, to=1.0, value=0.1, command=update_slider1, length=300)
        # slider1.pack(pady=5); label1 = ttk.Label(window, text="10%", background="#0A0A0A", foreground="white"); label1.pack()

        # slider2 = ttk.Scale(window, from_=0.1, to=1.0, value=1.0, command=update_slider2, length=300)
        # slider2.pack(pady=5); label2 = ttk.Label(window, text="100%", background="#0A0A0A", foreground="white"); label2.pack()
    ====================================================================
    """

    min_value = tk.DoubleVar(value="0.1")
    min_spinbox = ttk.Spinbox(from_=0.1, to=1.0, increment=0.1, textvariable=min_value, font=("monospace", 14, "bold"), width=4)
    min_spinbox.pack(pady=5)
    
    max_value = tk.DoubleVar(value="1.0")
    max_spinbox = ttk.Spinbox(from_=0.1, to=1.0, increment=0.1, textvariable=max_value, font=("monospace", 14, "bold"), width=4)
    max_spinbox.pack(pady=5)

    def create_entry(label_text, default_val):
        ttk.Label(window, text=label_text, background="#0A0A0A", foreground="white").pack(pady=(15, 0))
        e = tk.Entry(window, background="#1A1A1A", foreground="white", insertbackground="white", font=("monospace", 14, "bold"), justify='center')
        e.insert(0, default_val); e.pack(pady=5, padx=20, fill='x')
        return e

    entry_path = create_entry("File Path:", "portfolio.txt")
    entry_budget = create_entry("Budget:", "1000")
    entry_epochs = create_entry("Epochs:", "100")
    entry_poblation = create_entry("Poblation Size:", "100")
    entry_deselection = create_entry("Natural Deselection:", "75")

    btn = tk.Button(window, text="GO", bg="#1A1A1A", fg="white", font=("Segoe UI", 12, "bold"), width=30, cursor="hand2", command=start)
    btn.pack(pady=30)

    global listbox
    ttk.Label(window, text="Mejor combinación de Proyectos:", font=("monospace", 16, "bold"), background="#0A0A0A", foreground="white").pack(pady=20)
    listbox = tk.Listbox(window, width=60)
    listbox.pack(pady=5)

    return window

listbox = None
data, projs_dict = {}, {}
listbox = None
gui_active = False

if __name__ == '__main__':

    with Manager() as manager:
        
        shared_goat = manager.dict({
            'fitness': -1, 
            'epoch': -1, 
            'probability': -1, 
            'strength': -1, 
            'projects': ""
        })

        shared_history = manager.list()

        shared_lock = manager.Lock()

        # Correct amount of args
        if len(sys.argv) == 1:
            print("No arguments. Launching GUI...")
            gui(shared_goat, shared_history, shared_lock).mainloop()
        elif len(sys.argv) == 3:
            try:

                file_name = sys.argv[1]
                budget = float(sys.argv[2])
                with open(file_name, 'r', encoding='UTF-8') as f:
                    data = f.read().split('\n')
                    # remove empty lines
                    data = [line for line in data if line.strip()]

                    default_epochs = 100
                    default_poblation = 100
                    default_natural_deselection = 75
                    default_min_val = 0.1
                    default_max_val = 1.0

                    spawn_processes(
                        shared_goat,
                        shared_history,
                        shared_lock,
                        data,
                        budget,
                        default_epochs,
                        default_poblation,
                        default_natural_deselection,
                        default_min_val,
                        default_max_val
                    )


            except ValueError:
                print("[Error] Incorrect arguments. Launching GUI...")
                print('\n\tArguments: myprogram.py "file_path.txt" 1000')
                gui(shared_goat, shared_history, shared_lock).mainloop()
            except FileNotFoundError:
                print(f"File {file_name} doesn't exist")
            except Exception as e:
                print(f"Uncaught error: {e}")
        else:
            gui(shared_goat, shared_history, shared_lock).mainloop()
