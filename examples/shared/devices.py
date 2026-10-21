"""Aggregate programs run device by device on a network whose links change every round."""

from diffield.decentralized import build_runtimes, require_symmetric
from diffield.decentralized.topology import all_in_edges


def nbytes(tensors):
    """The bytes of the elements of ``tensors``, as sent on the wire without framing."""
    return sum(t.numel() * t.element_size() for t in tensors)


def ranges(edge_index, positions):
    """The length of every link between devices at ``positions``."""
    a, b = edge_index
    return (positions[a] - positions[b]).norm(dim=-1)


class DeviceNetwork:
    """One :class:`~diffield.decentralized.DeviceRuntime` per node, exchanging state bundles.

    A device that does not run in a round keeps its last output in ``outputs``, and every
    device keeps the last bundle heard from each of its current neighbours. Which devices run,
    and in which order, is up to the caller, and so is any randomness behind that choice.

    Args:
        program: The aggregate program every device runs.
        edge_index: The initial links, ``[2, E]``.
        num_nodes: The number of devices.
        outputs: ``[num_nodes, ...]``; row ``i`` receives the output of device ``i``.
        positions: Where the devices are, to measure their links; without it every link is 1.
        signals: Global sensor fields, which every device slices to its neighbourhood.
    """

    def __init__(self, program, edge_index, num_nodes, outputs, *, positions=None, signals=None):
        self.devices = build_runtimes(
            edge_index=edge_index,
            num_nodes=num_nodes,
            program=program,
            signals=signals or {},
            edge_weight=None if positions is None else ranges(edge_index, positions),
        )
        self.outputs = outputs
        self._published = {node: {} for node in self.devices}
        self._mailboxes = {node: {} for node in self.devices}

    def move(self, edge_index, *, positions=None, signals=None):
        """Give every device its current links and sensors; removed links stop delivering."""
        require_symmetric(edge_index)
        weights = None if positions is None else ranges(edge_index, positions)
        for node, links in enumerate(all_in_edges(edge_index, len(self.devices), weights)):
            device = self.devices[node]
            device.sync_topology(*links, signals=signals)
            mailbox = self._mailboxes[node]
            self._mailboxes[node] = {j: v for j, v in mailbox.items() if j in device.neighbor_ids}

    def run(self, nodes, *, barrier=True):
        """Run ``nodes`` in order, and return the bytes they send.

        Under a barrier, devices read the bundles published in the previous round and deliver
        theirs after the round; otherwise each reads its mailbox and delivers at once.
        """
        pending, sent = {}, 0
        for node in nodes:
            device = self.devices[node]
            received = self._published if barrier else self._mailboxes[node]
            keys = {key for other in device.neighbor_ids for key in received.get(other, {})}
            inbox = {
                key: [received.get(other, {}).get(key) for other in device.neighbor_ids]
                for key in keys
            }
            self.outputs[node], bundle = device.step(inbox)
            pending[node] = bundle
            sent += nbytes(bundle.values()) * len(device.neighbor_ids)
            if not barrier:
                for other in device.neighbor_ids:
                    self._mailboxes[other][node] = bundle
        self._published.update(pending)
        if barrier:
            for node, bundle in pending.items():
                for other in self.devices[node].neighbor_ids:
                    self._mailboxes[other][node] = bundle
        return sent

    @property
    def state_bytes(self):
        """The bytes of the bundles the devices published last."""
        return sum(nbytes(bundle.values()) for bundle in self._published.values())
