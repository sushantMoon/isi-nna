import _pickle as pickle
import os


class SaveLoad():
    def __init__(self):
        pass

    @classmethod
    def save_object(cls, file_path, python_object):
        """saves the Python Obejct to the disk at the `file_path`

        Arguments:
            file_path {str} -- path to a file where
            we would be storing the Python Obejct
            python_object {Python Obejct} -- Python Obejct which
            we want to save
        """
        print("Saving Python Obejct to {}".format(file_path))
        with open(file_path, 'ab') as pklfile:
            pickle.dump(python_object, pklfile)
        print("Python Obejct Saved at {}".format(file_path))

    @classmethod
    def load_object(cls, saved_object_path):
        """loads the python object trained earlier

        Arguments:
            saved_object_path {str} -- path to the saved python object

        Returns:
            Python Object -- returns the python object
        """
        assert os.path.exists(saved_object_path), (
            "Check the provided path for the saved python obejct."
        )
        with open(saved_object_path, 'rb') as pklfile:
            python_object = pickle.load(pklfile)
        return python_object
