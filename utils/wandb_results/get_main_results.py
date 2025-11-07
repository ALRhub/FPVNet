from collections import defaultdict
from IPython import embed
import wandb2numpy
import numpy as np

import os, sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


config = {
    "local": {
        "entity": "tiger_or_cat",
        "project": "atalay_robocasa_final",
        "groups": "",
        "fields": "",
        "runs": ["all"],
        "config": "",
        "output_path": "",
    }
}

if __name__ == "__main__":

    agent_names = [
        # "dp3",
        # "diffuser_actor",
        # "point_mlp",
        # "rgb",
        # "point_mlp_resnet",
        # "point_mlp_resnet",
        # "point_mlp_resnet_pc_cond",
        # "point_mlp_resnet_pc_cond_transformer",
        # "point_mlp_resnet_img_cond",
        # "point_mlp_resnet_img_cond_transformer",
        # "point_mlp_resnet_local_img_cond_transformer",
        # "point_mlp_resnet_local8x8_img_cond_transformer",
        # "point_mlp_resnet_local8x8_img_cond_transformer_l512",
        # "point_mlp_resnet_local_img_cond_transformer",
        # "sugar_resnet18_local_img_cond_transformer",
        # "sugar_ft2_resnet18_local_img_cond_transformer",
        # "point_mlp_resnet_local8x8_img_cond_transformer",
        # "point_mlp_resnet_local8x8_img_cond_transformer_l512",
        # "sugar_resnet18_local8x8_img_cond_transformer",
        # "sugar_resnet18_local8x8_img_cond_transformer",
        # "sugar_ft2_resnet18_local8x8_img_cond_transformer_l512_e128",
        # "sugar_ft5_resnet18_local8x8_img_cond_transformer_l512"
        # "sugar_ft2_resnet18_local8x8_img_cond_transformer_l256_e128",
        # "sugar_ft2_resnet18_local8x8_img_cond_transformer_l256_e256",
        # "sugar_ft2_resnet18_local8x8_img_cond_transformer_l512_e128",
        # "sugar_ft2_resnet18_local8x8_img_cond_transformer_l512_e256",
        # "point_mlp_resnet_local_img_cond_transformer",
        # "point_mlp_resnet34_local_img_cond_transformer",
        # "sugar_resnet18_local_img_cond_transformer",
        # "sugar_resnet34_local_img_cond_transformer",
        "sugar_ft2_resnet18_local8x8_img_cond_transformer_l512_e128",
        "sugar_ft5_resnet18_local8x8_img_cond_transformer_l512",
    ]

    # agent_names = [
    #     "global_point_mlp_resnet_local_img_cond_transformer_with_pos_emb",
    #     "global_point_mlp_resnet_local_img_cond_transformer"
    # ]

    env_groups = {
        "bc_pnp1": [
            "PnPCounterToCab_average_success",
            "PnPCabToCounter_average_success",
            "PnPCounterToSink_average_success",
            "PnPSinkToCounter_average_success",
        ],
        "bc_pnp2": [
            "PnPCounterToMicrowave_average_success",
            "PnPMicrowaveToCounter_average_success",
            "PnPStoveToCounter_average_success",
            "PnPCounterToStove_average_success",
        ],
        "bc_doors": [
            "OpenSingleDoor_average_success",
            "CloseSingleDoor_average_success",
            "OpenDoubleDoor_average_success",
            "CloseDoubleDoor_average_success",
        ],
        "bc_drawer": ["CloseDrawer_average_success", "OpenDrawer_average_success"],
        "bc_stove": ["TurnOnStove_average_success", "TurnOffStove_average_success"],
        "bc_sink_minmax": [
            "TurnOnSinkFaucet_average_success",
            "TurnOffSinkFaucet_average_success",
            "TurnSinkSpout_average_success",
        ],
        "bc_buttons": [
            "CoffeePressButton_average_success",
            "TurnOffMicrowave_average_success",
            "TurnOnMicrowave_average_success",
        ],
        "bc_coffee": [
            "CoffeeSetupMug_average_success",
            "CoffeeServeMug_average_success",
        ],
    }

    # embed_dims = [128, 256]

    agent_success_rates = defaultdict(list)

    for env_group, fields in env_groups.items():
        env_success_rates = defaultdict(list)
        for field in fields:
            strings = []
            for agent_name in agent_names:
                config["local"]["groups"] = [env_group]
                config["local"]["fields"] = [field]

                config["local"]["config"] = {
                    "agent_name": {"values": [agent_name]},
                    # "n_embd": {"values": [256 if agent_name in ["sugar_ft2_resnet18_local_img_cond_transformer", "rgb"] else 128]},
                    "n_embd": {"values": [128]},
                }

                with HiddenPrints():
                    data_dict, config_list = wandb2numpy.export_data(config)

                if not data_dict:
                    number_string = f"& "
                    strings.append(number_string)
                    continue

                success = data_dict["local"][field].squeeze()

                if success.size == 0:
                    number_string = f"& "
                    strings.append(number_string)
                else:
                    # env_success_rates[f"{agent_name}_{embed_dim}"].append(success)
                    # agent_success_rates[f"{agent_name}_{embed_dim}"].append(success)
                    env_success_rates[agent_name].append(success)
                    agent_success_rates[agent_name].append(success)
                    number_string = f"& ${success:.2f}$"
                    strings.append(number_string)

            # print(
            #     bcolors.WARNING
            #     + f"-------------------{field}-------------------"
            #     + bcolors.ENDC
            # )
            # for string in strings:
            #     print(string)
        print(
            bcolors.WARNING
            + f"-------------------{env_group}-------------------"
            + bcolors.ENDC
        )
        for agent, success_rates in env_success_rates.items():
            success_rates = np.array(success_rates)
            avg_success = np.mean(success_rates)
            # print(f"& ${avg_success:.4f}$")
            print(f"{agent} & ${avg_success:.4f}$")

    print(
        bcolors.OKGREEN
        + f"-------------------Average Success Rates-------------------"
        + bcolors.ENDC
    )
    strings = []
    for agent, success_rates in agent_success_rates.items():
        if len(success_rates) != 24:
            # if len(success_rates) != 12:
            # if False:
            print(f"& ")
        else:
            success_rates = np.array(success_rates)
            avg_success = np.mean(success_rates)

            print(f"& ${avg_success:.4f}$")
            # print(f"{agent} & ${avg_success:.4f}$")
