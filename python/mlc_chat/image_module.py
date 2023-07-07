"""Image module for MLC chat."""
#! pylint: disable=unused-import, invalid-name
import tvm
import tvm._ffi.base


class ImageModule:
    def __init__(self, target: str = "cuda", device_id: int = 0):
        r"""Initialize an image module.

        Parameters
        ----------
        target : str
            The target device type.
        device_id : int
            The device id.
        """
        fcreate = tvm.get_global_func("mlc.llm_image_mod_create")
        assert fcreate is not None
        if target == "cuda":
            self.device = tvm.cuda(device_id)
        elif target == "metal":
            self.device = tvm.metal(device_id)
        elif target == "vulkan":
            self.device = tvm.vulkan(device_id)
        else:
            raise ValueError("device type not supported yet")
        device_type = self.device.device_type
        image_mod = fcreate(device_type, device_id)

        self.reload_func = image_mod["reload"]
        self.embed_func = image_mod["embed"]
        self.reset_image_mod_func = image_mod["reset_image_mod"]
        self.runtime_stats_text_func = image_mod["runtime_stats_text"]
        self.reset_runtime_stats_func = image_mod["reset_runtime_stats"]

    def reload(self, lib: str, model_path: str):
        r"""Reload the image module from the given library and model path.

        Parameters
        ----------
        lib : str
            The library path.
        model_path : str
            The model path.
        """
        self.reload_func(lib, model_path)

    def embed(
        self,
        image: tvm.runtime.NDArray,
    ):
        r"""Given an image of type NDArray, get the embedding of the image.

        Parameters
        ----------
        image : tvm.runtime.NDArray
            The user uploaded image.
        """
        return self.embed_func(image)

    def reset_image_mod(self):
        r"""Reset the image module, clear its performance record.

        Note
        ----
        The model remains the same after :func:`reset_image_mod`.
        To reload module, please use :func:`reload` instead.
        """
        self.reset_chat_func()

    def runtime_stats_text(self) -> str:
        r"""Get the runtime stats text (image encoding speed).

        Returns
        -------
        stats : str
            The runtime stats text.
        """
        return self.runtime_stats_text_func()

    def reset_runtime_stats(self):
        r"""Reset the runtime stats."""
        self.reset_runtime_stats_func()
