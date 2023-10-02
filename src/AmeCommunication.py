"""
This modules defines two classes that make it possible to communicate with AMESim submodels
"""
import sys
import os

if sys.platform.startswith("linux"):
    path_to_append = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "lnx_x64")
    )
    sys.path.append(path_to_append)
else:
    path_to_append = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "win64_vc140")
    )
    sys.path.append(path_to_append)

print("PATH TO APPEND=>")
print(path_to_append)
print("===" * 14)
print("SYS PATH=>")
print(sys.path)


import traceback

try:
    import src.binding_amecommunication as binding_amecommunication
except Exception as e:
    print(e.__cause__)
    traceback.print_exc()

import unittest


class AmeExchanger:
    """
    base virtual class
    """

    def init(self):
        raise NotImplemented

    def exchange(self):
        raise NotImplemented

    def close(self):
        raise NotImplemented

    def __del__(self):
        self.close()


class AmeSocket(AmeExchanger):
    """
    a class to communicate with AMESim submdel via TCP/IP
    """

    def __init__(self):
        self.sock_id = 0
        self.connected = False

    def init(self, isserver, server_name, server_port, nb_input, nb_output):
        """
        initialize connection

        ``isserver``: boolean that indicates if the connection should be opened as server or client
        ``server_name``: server host name. (could also be an ip)
        ``server_port``: server port
        ``nb_input``: number of values that will be send from our side to the other side.
                          This number includes the mandatory values next_meeting and local_time.
                          That's why nb_input can't be less than 2.
        ``nb_output``: number of values that will be received from the other side.
                            This number also include the mandatory values next_meeting and local_time.
                            That's why nb_output can't be less than 2.
        """
        if isserver:
            isserver_int = 1
        else:
            isserver_int = 0

        ret = binding_amecommunication.sockinit(
            isserver_int, server_name, server_port, nb_input, nb_output
        )
        if ret[0] != 0:
            raise RuntimeError("unable to initialize socket (" + ret[0].__str__() + ")")
        self.sock_id = ret[1]
        self.connected = True

    def exchange(self, input):
        """
        exchange values

        ``input``: list of values to send.
        return the received values
        """
        if self.sock_id == 0:
            raise RuntimeError("uninitialized sharedmem")
        ret = binding_amecommunication.sockexchange(self.sock_id, input)
        if ret[0] != 0:
            raise RuntimeError("unable to exchange values (" + ret[0].__str__() + ")")
        return ret[1]

    def close(self):
        """
        close connection
        """
        if not self.connected:
            return
        if self.sock_id == 0:
            raise RuntimeError("uninitialized sharedmem")
        ret = binding_amecommunication.sockclose(self.sock_id)
        if ret != 0:
            raise RuntimeError("failed to close socket (" + ret[0].__str__() + ")")
        self.sock_id = 0
        self.connected = False


class AmeSharedmem(AmeExchanger):
    """
    a class to communicate with AMESim submdel via shared memory
    """

    def __init__(self):
        self.shm_id = 0
        self.connected = False

    def init(self, ismaster, name, nb_input, nb_output):
        """
        initialize connection

        ``ismaster``: boolean that indicates if the connection should be opened as master or slave
        ``name``: identifier for shared memory
        ``nb_input``: number of values that will be send from our side to the other side.
                          This number includes the mandatory values next_meeting and local_time.
                          That's why nb_input can't be less than 2.
        ``nb_output``: number of values that will be received from the other side.
                            This number also include the mandatory values next_meeting and local_time.
                            That's why nb_output can't be less than 2.
        """
        if ismaster:
            ismaster_int = 1
        else:
            ismaster_int = 0

        ret = binding_amecommunication.shminit(ismaster_int, name, nb_input, nb_output)
        if ret[0] != 0:
            raise RuntimeError(
                "unable to initialize shared memory (" + ret[0].__str__() + ")"
            )
        self.shm_id = ret[1]
        self.connected = True

    def exchange(self, input):
        """
        exchange values

        ``input``: list of values to send.
        return the received values
        """
        if self.shm_id == 0:
            raise RuntimeError("uninitialized sharedmem")
        ret = binding_amecommunication.shmexchange(self.shm_id, input)
        if ret[0] != 0:
            raise RuntimeError("unable to exchange values (" + ret[0].__str__() + ")")
        return ret[1]

    def close(self):
        """
        close connection
        """
        if not self.connected:
            return
        if self.shm_id == 0:
            raise RuntimeError("uninitialized sharedmem")
        ret = binding_amecommunication.shmclose(self.shm_id)
        if ret != 0:
            raise RuntimeError(
                "failed to close shared memory (" + ret[0].__str__() + ")"
            )
        self.shm_id = 0
        self.connected = False


class TestSequenceFunctions(unittest.TestCase):
    """very basic test class for AmeSocket and AmeSharedmem"""

    def testAmeSocket(self):
        """test that AmeSocket is known"""
        a = AmeSocket()
        self.assert_(a is not None)
        a.close()

    def testAmeSharedmem(self):
        """test that AmeSharedmem is known"""
        a = AmeSharedmem()
        self.assert_(a is not None)
        a.close()


if __name__ == "__main__":
    unittest.main()
