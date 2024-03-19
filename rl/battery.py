import numpy as np
import pybamm
import numbers
import gymnasium as gym
from gymnasium import spaces
import os


def make_new_model(model, param, disc, update_input):
    model1 = model.new_copy()
    param1 = param.copy()

    if update_input is not None:
        param1.update(update_input)
    else:
        pass
    model1 = param1.process_model(model1, inplace=False)
    built_model = disc.process_model(model1, inplace=False, check_model=True)
    return built_model

def make_new_model_with_parameters(model, param, disc, update_input):
    model1 = model.new_copy()
    param1 = param.copy()

    if update_input is not None:
        param1.update(update_input)
    else:
        pass
    model1 = param1.process_model(model1, inplace=False)
    built_model = disc.process_model(model1, inplace=False, check_model=True)
    return built_model


def update_model_step(inputparam, model, param, disc, solutions):
    model = make_new_model(model, param, disc, inputparam)
    solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)

    step_solution = solver.step(solutions[-1].last_state,
                    model,
                    1,
                    npts=100,
                    save=False,)
    return step_solution

def update_input(current):
    if isinstance(current, np.ndarray) and current.size == 1:
        current = current.item()  # Convert single-element array to scalar
    elif not isinstance(current, numbers.Number):
        print("Type of current:", type(current))
        raise TypeError("current must be a number or a single-element numpy array")
    
    update_input = {
        "Current function [A]": current,
    }
    return update_input


class BatteryEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):

        # Observations are dictionaries
        #model.variable_names has 468 variables
        self.observation_space = spaces.Dict(
        {
            "Current function [A]": spaces.Box(-4, 4, shape=(1,), dtype=np.float64),
            "Battery voltage known[V]": spaces.Box(0, 5, shape=(1,), dtype=np.float64),
            "Battery voltage unknown[V]": spaces.Box(0, 5, shape=(1,), dtype=np.float64)
        }
        )

        # we have 1 actions, current
        self.action_space = spaces.Box(-5,5, shape=(), dtype=np.float16)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None


    def reset(self,options=None,seed=None):
        self.reward = 0
        self.terminated = False
        self.truncated = False
        options_known = {"thermal": "lumped"}
        # options_known = {"number of rc element": 2}
        options_unknown = None

        self.model_known = pybamm.lithium_ion.DFN(options=options_known)
        self.model_unknown = pybamm.lithium_ion.DFN(options=options_unknown)
        # self.model_known = pybamm.equivalent_circuit.Thevenin(options=options_known)
        # self.model_unknown = pybamm.equivalent_circuit.Thevenin(options=options_unknown)
        self.params = pybamm.ParameterValues("Ecker2015").copy()

        self.params["Current function [A]"] = 0.0
        self.params.process_model(self.model_known)

        #setting geometry
        geometry = self.model_known.default_geometry
        submesh_types = self.model_known.default_submesh_types
        var_pts = self.model_known.default_var_pts
        spatial_methods = self.model_known.default_spatial_methods
        self.params.process_geometry(geometry)
        #setting mesh

        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        self.disc = pybamm.Discretisation(mesh, spatial_methods)
        init_model = self.disc.process_model(self.model_known, inplace=False, check_model=True)
        self.solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)

        #set initial solution
        t_eval = np.linspace(0, 1e-10, 2)
        sim = pybamm.Simulation(init_model,parameter_values=self.params, solver=self.solver)
        self.initial_solution = sim.solve(t_eval=t_eval)

        self.solutions_known = []
        self.solutions_known += [self.initial_solution]
        self.solutions_unknown = []
        self.solutions_unknown += [self.initial_solution]
        #setting initial conditions
        self.observation = {
        "Battery voltage known[V]": np.array([3.7]),
        "Battery voltage unknown[V]": np.array([3.7]),
        "Current function [A]": np.array([0.0]),
        }

        info = {"model_known": self.model_known, "model_unknown": self.model_unknown, "param": self.params, "disc": self.disc}
        return self.observation, info
    

    def step(self, action):

        self.solutions_known += [update_model_step(update_input(action), self.model_known, self.params, self.disc, self.solutions_known)]
        self.solutions_unknown += [update_model_step(update_input(action), self.model_unknown, self.params, self.disc, self.solutions_unknown)]

        voltage_known = self.solutions_known[-1]["Battery voltage [V]"].entries[-1]
        voltage_unknown = self.solutions_unknown[-1]["Battery voltage [V]"].entries[-1]

        # self.observation = {
        # "Battery voltage known[V]": np.array([voltage_known]),
        # "Battery voltage unknown[V]": np.array([voltage_unknown]),
        # "Current function [A]": action
        # }

        #dense reward
        rmse = np.sqrt(np.mean((voltage_known - voltage_unknown)**2))
        self.reward += -rmse

        info = {"model_known": self.model_known, "model_unknown": self.model_unknown, "param": self.params, "disc": self.disc}
        return self.observation, self.reward, self.terminated, self.truncated, info
    
    
    def render(self):
        output_variables = [
        "Battery voltage [V]"]
        plot = pybamm.QuickPlot(self.solutions_unknown, output_variables)
        plot.dynamic_plot()

    def close(self):
        pass

