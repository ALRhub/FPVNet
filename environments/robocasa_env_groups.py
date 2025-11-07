ENV_GROUPS = {
    "pnp1": [
        "PnPCounterToCab",
        "PnPCabToCounter",
        "PnPCounterToSink",
        "PnPSinkToCounter",
    ],
    "pnp2": [
        "PnPCounterToMicrowave",
        "PnPMicrowaveToCounter",
        "PnPCounterToStove",
        "PnPStoveToCounter",
    ],
    "doors": ["OpenSingleDoor", "CloseSingleDoor", "OpenDoubleDoor", "CloseDoubleDoor"],
    "drawer": ["OpenDrawer", "CloseDrawer"],
    "sink": ["TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout"],
    "stove": ["TurnOnStove", "TurnOffStove"],
    "coffee": ["CoffeeSetupMug", "CoffeeServeMug"],
    "buttons": ["CoffeePressButton", "TurnOnMicrowave", "TurnOffMicrowave"],
}