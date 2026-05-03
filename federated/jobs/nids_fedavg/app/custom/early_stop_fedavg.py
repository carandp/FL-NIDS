from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.fuel.utils.log_utils import center_message


class EarlyStopFedAvg(FedAvg):
    def __init__(self, *args, early_stop_min_clients=None, **kwargs):
        super().__init__(*args, **kwargs)
        if early_stop_min_clients is None:
            early_stop_min_clients = self.num_clients
        self.early_stop_min_clients = early_stop_min_clients
        self._early_stopped_clients = set()

    def _update_early_stopped_clients(self, results):
        for result in results:
            if not result or not result.meta:
                continue
            if result.meta.get("early_stopped"):
                client_name = result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN)
                if client_name:
                    self._early_stopped_clients.add(client_name)

    def _sample_active_clients(self, num_clients: int):
        clients = [client.name for client in self.engine.get_clients()]
        active_clients = [c for c in clients if c not in self._early_stopped_clients]

        if not active_clients:
            return []

        if num_clients and num_clients != len(active_clients):
            self.info(
                f"Ignoring num_clients ({num_clients}); using all ({len(active_clients)}) active clients."
            )

        self.info(f"Sampled active clients: {active_clients}")
        return active_clients

    def run(self) -> None:
        self.info(center_message("Start FedAvg (early-stop enabled)."))

        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(center_message(message=f"Round {self.current_round} started.", boarder_str="-"))

            model.current_round = self.current_round

            clients = self._sample_active_clients(self.num_clients)
            if not clients:
                self.info("No active clients remaining; ending run.")
                break

            results = self.send_model_and_wait(targets=clients, data=model)
            if not results:
                self.info("No client results received; ending run.")
                break

            self._update_early_stopped_clients(results)
            active_results = [r for r in results if not r.meta.get("early_stopped")]

            if len(self._early_stopped_clients) >= self.early_stop_min_clients:
                self.info(center_message("All clients early-stopped. Ending run."))
                break

            if not active_results:
                self.info("All client updates are early-stopped; ending run.")
                break

            aggregate_results = self.aggregate(active_results, aggregate_fn=self.aggregate_fn)
            model = self.update_model(model, aggregate_results)
            self.save_model(model)

        self.info(center_message("Finished FedAvg."))
