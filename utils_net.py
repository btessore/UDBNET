import os
import numpy as np
import warnings
from contextlib import nullcontext
import h5py
from sklearn.decomposition import PCA
from pdf2image import convert_from_path, convert_from_bytes
import cv2
import io
from typing import List


def convert_pdf_to_images(
    pdf_path: str, dpi: int = 300, cvtColor_enum: int = cv2.COLOR_BGR2GRAY
) -> List[np.ndarray]:
    """Convert a PDF file to a list of images in a format compatible with OpenCV.

    Args:
        pdf_path (str): The path to the PDF file.
        dpi (int, optional): _description_. Defaults to 100.
        cvtColor_enum (int, optional): An enum option of reading with cv2. (Default BGR to GRAY).

    Returns:
        List[np.ndarray]:  Dots per inch for the converted images
    """
    # Convert PDF to list of PIL Image objects
    pil_images = convert_from_path(pdf_path, dpi=dpi)
    pil_img = pil_images[0]

    # Convert PIL Images to OpenCV format
    opencv_image = cv2.cvtColor(np.array(pil_img), cvtColor_enum)

    return opencv_image


def read_image(
    f, dpi: int = 300, imread_enum: int = cv2.IMREAD_GRAYSCALE
) -> np.ndarray:
    """Read an image which is either a file object or a path
    to a an image file."""
    cvtColor_enum = (
        imread_enum if imread_enum != cv2.IMREAD_GRAYSCALE else cv2.COLOR_BGR2GRAY
    )
    if isinstance(f, io.IOBase):
        # file object
        if f.name.endswith((".pdf", ".PDF")):
            im = cv2.cvtColor(
                np.array(convert_from_bytes(f.read(), dpi=dpi)[0]), cvtColor_enum
            )
        else:
            tmp = np.frombuffer(f.read(), np.uint8)
            im = cv2.imdecode(tmp, imread_enum)
        f.seek(0)
    else:
        # path
        if f.endswith((".pdf", ".PDF")):
            im = convert_pdf_to_images(f, dpi=dpi, cvtColor_enum=cvtColor_enum)
        else:
            im = cv2.imread(f, imread_enum)
    return im


# TO DO: add a test on the size of the image divisible by patch size
def extract_patch(x, ip, patch_size, step_size, patch_overlap_size):
    """Extract the SINGLE ip patch of image x.

    The patch are ordered by columns (from left to right), then
    by row (from top to bottom).
    The index idx of the patch pertaining to a given image is
    ip = j * N_col + i, with (i, j) indexes of columns and rows respectively.
    From ip, i is given by ip % N_col and j by (ip - i) / N_col.

    Args:
        x (ndarray): image
        ip (int): index of a patch relative to image x.

    Returns:
        ndarray: the ip patch of image x.
    """
    # Number of columns in the grid of patch on image x
    N_patch_col = (x.shape[0] - patch_overlap_size[0]) // step_size[0]
    # N_patch_row = (x.shape[1] - patch_overlap_size[1]) // step_size[1]

    # index of the column ip is referring to
    i = ip % N_patch_col
    # index of the row ip is reffering to
    j = int((ip - i) / N_patch_col)

    start_i = i * step_size[0]
    start_j = j * step_size[1]
    end_i = start_i + patch_size[0]
    end_j = start_j + patch_size[1]

    return x[start_i:end_i, start_j:end_j]


def pad_func(im, shape, rule, color=False):
    if rule == "tile":
        pad_im = (
            np.tile(im, (2, 2))[: shape[0], : shape[1]]
            if not color
            else np.tile(im, (2, 2, 1))[: shape[0], : shape[1]]
        )
    elif rule in ["symmetric", "constant"]:
        pH = shape[0] - im.shape[0]
        pW = shape[1] - im.shape[1]
        pad_shape = ((0, pH), (0, pW)) if not color else ((0, pH), (0, pW), (0, 0))
        if rule == "symmetric":
            pad_im = np.pad(im, pad_shape, mode=rule)
        else:
            pad_im = np.pad(
                im,
                pad_shape,
                mode=rule,
                constant_values=255,
            )
    else:
        raise ValueError("Unkown value for padding rule %s" % rule)

    # if the color enum is wrong it could lead to false shape after padding!
    if color:
        assert pad_im.shape[-1] == 3, " Error while padding color image!"
    return pad_im


class Patch:
    def __init__(self, size=0, step=None, padding="tile"):
        self.use = size > 0
        self.rule = padding
        # Let the possibility of rectangular patches
        self.size = size
        if isinstance(self.size, int):
            self.size = (self.size, self.size)

        self.step = step
        # By default no overlaps
        if step is None:
            self.step = self.size
        # Again, the overlap could be rectangular
        if isinstance(self.step, int):
            self.step = (self.step, self.step)

        # size of the overlap between patches
        self.overlap_size = (
            self.size[0] - self.step[0],
            self.size[1] - self.step[1],
        )
        # in other works, the step cannot be larger than the patch_size, to not create gap
        if self.overlap_size[0] >= self.size[0] or self.overlap_size[1] >= self.size[1]:
            raise ValueError(
                "Overlapping size cannot be greater or equal than patch size!"
            )

    def patch(self, x, ip):
        """Extract the idx patch of image x.

        The patch are ordered by columns (from left to right), then
        by row (from top to bottom).
        The index idx of the patch pertaining to a given image is
        ip = j * N_col + i, with (i, j) indexes of columns and rows respectively.
        From ip, i is given by ip % N_col and j by (ip - i) / N_col.

        Args:
            x (ndarray): image
            ip (int): index of a patch relative to image x.

        Returns:
            ndarray: the ip patch of image x.
        """
        return extract_patch(x, ip, self.size, self.step, self.overlap_size)

    def count_patch(self, x):
        """Counts number of patches per images.

        Args:
            x (ndarray): input image

        Returns:
            int: number of patches in image x
        """
        # H, W = self.__pimage_shape__(x)
        # includes padding
        # N_col, N_row = (
        #     (H - self.patch_overlap_size[0]) // self.step_size[0],
        #     (W - self.patch_overlap_size[1]) // self.step_size[1],
        # )
        # Number of patches along each dimension should take into account the overlap
        n_col, n_row = (
            (x.shape[0] - self.overlap_size[0]) // self.step[0],
            (x.shape[1] - self.overlap_size[1]) // self.step[1],
        )
        return n_col * n_row

    def padded_shape(self, x):
        """Return the shape of image x that is divisible by patch size"""
        return self.size[0] * (int(x.shape[0] / self.size[0]) + 1) if x.shape[
            0
        ] % self.size[0] else x.shape[0], self.size[1] * (
            int(x.shape[1] / self.size[1]) + 1
        ) if x.shape[1] % self.size[1] else x.shape[1]

    def load_image(self, f, enum=1, dpi=200):
        d = read_image(f, imread_enum=enum, dpi=dpi)
        if enum > 0:
            assert d.shape[-1] == 3, "Error, image is not read as colored image"
        if self.use:
            H, W = self.padded_shape(d)
            d = pad_func(d, (H, W), self.rule, color=enum > 0)
        return d


def store_patches(filename, files, patch, enum=1, dpi=200):
    """write to disc all patches associted to a given list of image files"""
    if not patch.use:
        raise ValueError("Err store_patches!")
    with h5py.File(filename, "w") as cache:
        offset = 0
        last_patch = 0
        for i in range(len(files)):
            d = patch.load_image(files[i], enum=enum, dpi=dpi)
            n_patch = patch.count_patch(d)
            i0 = 0 if i == 0 else last_patch
            i1 = i0 + n_patch
            ip = 0
            for idx in range(i0, i1):
                p = patch.patch(d, ip)
                cache.create_dataset(f"{idx}", data=p, compression="gzip")
                ip += 1

            last_patch = n_patch + offset
            offset += n_patch

    return offset


# takes old hand-written and clean CAO
# count the number of patches or number of individual CAO and hand-written
# The number of CAO must be chosen to match the number of hand-written enventually attributing
# the same hand-written to several CAO. Select randomly how this attribution is done.
# output pairs of (hand-written, CAO).
# The batch outputed by the dataloader should be (2,B,C,H,W)
# containing each batch for each pair


class Dataset:
    """Data generator for machine learning training"""

    def __init__(
        self,
        hdw_files,
        cad_files,
        patch=None,
        transform=None,
        eps=0.9,
        limit_memory=False,
        cache_file=None,
        pca_fit=False,
        seed=42,
    ):
        """Initialisation of the data generator."""
        self.hdw_files = hdw_files
        self.n_hdw = len(self.hdw_files)
        self.cad_files = cad_files
        self.n_cad = len(self.cad_files)
        self.eps = eps
        self.transform = transform
        self.cache_file = cache_file
        self.limit_memory = limit_memory
        # make colored image gray-scale with pca
        self.pca_fit = pca_fit
        # random creation of pairs
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.frac = len(self.cad_files) / len(self.hdw_files)
        self.Patch = patch

        self.build_cache_database = limit_memory  # init
        if self.limit_memory:
            self.cache_file = cache_file if cache_file else ".cache.tmp"

            if os.path.exists(self.cache_file):
                warnings.warn(
                    "Found an existing database for %s. Using it !" % self.cache_file
                )

                self.build_cache_database = False
                with h5py.File(self.cache_file, "r") as self.cache_fileobj:
                    # (inputs + targets) / 2
                    self.num_samples = len(self.cache_fileobj) // 2
                return
            else:
                print("Building cache database while processing data...")
        # Database is non-existant yet, creating it
        # first write all patches to tmp file if needed
        if self.Patch:
            print(" ** Building tmp cache for patches...")
            n_patches_hdw = store_patches(
                ".hdw.tmp",
                self.hdw_files,
                self.Patch,
                enum=1 if self.pca_fit else 0,
                dpi=200,
            )
            self.hdw_ind = np.arange(n_patches_hdw)
            n_patches_cad = store_patches(
                ".cad.tmp", self.cad_files, self.Patch, enum=0, dpi=200
            )
            self.cad_ind = np.arange(n_patches_cad)
            print(" ** done.")
            # redefining fraction of data pairs
            print(n_patches_cad, n_patches_hdw)
            print("cad-to-hdw patches ratio:", n_patches_cad / n_patches_hdw)
            print("self.frac", self.frac)
            self.frac = n_patches_cad / n_patches_hdw
            self.f_hdw = h5py.File(".hdw.tmp", "r")
            self.f_cad = h5py.File(".cad.tmp", "r")
        else:
            self.cad_ind = np.arange(self.n_cad)
            self.hdw_ind = np.arange(self.n_hdw)
        # redistribute pairs
        self.rng.shuffle(self.cad_ind)
        self.rng.shuffle(self.hdw_ind)
        print("cad-to-hdw ratio:", self.frac)
        if self.frac >= 1:  # more cad than hdw, replicated hdw
            self.hdw_ind = np.tile(self.hdw_ind, int(np.ceil(self.frac)))[
                : len(self.cad_ind)
            ]
        else:  # more hdw than cad, replicate cad
            self.cad_ind = np.tile(self.cad_ind, int(np.ceil(1 / self.frac)))[
                : len(self.hdw_ind)
            ]
        assert len(self.hdw_ind) == len(self.cad_ind), (
            "Number of cad and hdw files mistmatch!"
        )

        # same number of elements at that point in cad and hdw
        self.num_samples = len(self.hdw_ind)
        self.num_pairs = self.num_samples
        # read the data directly from file if not use patch
        # otherwise the data patched are stored in a binary file
        # but then store the data in memory or write the cache file for the database
        # as usual.

        self.data_hdw = []
        self.data_cad = []

        # loop over all pairs of data and open a cache file if needed.
        # WARNING: here the loop goes directly on the patches and not on original images
        # so the database is easily constructed (no need to update, add2cache, nextData etc..)
        with (
            h5py.File(self.cache_file, "w")
            if self.build_cache_database
            else nullcontext() as self.cache_fileobj
        ):
            for i in range(self.num_samples):
                if self.Patch:
                    hdw = np.array(self.f_hdw[str(self.hdw_ind[i])], dtype=np.uint8)
                    cad = np.array(self.f_cad[str(self.cad_ind[i])], dtype=np.uint8)
                else:
                    hdw = read_image(
                        self.hdw_files[self.hdw_ind[i]],
                        imread_enum=1 if self.pca_fit else 0,
                        dpi=200,
                    )
                    if self.pca_fit:
                        assert hdw.shape[-1] == 3, "Error reading colored scale image"
                    cad = read_image(
                        self.cad_files[self.cad_ind[i]], imread_enum=0, dpi=200
                    )

                # create gray-scale from colored-scaled using PCA
                if self.pca_fit:
                    pca = PCA(1)
                    # Note: absolute value
                    hdw = abs(
                        pca.fit_transform(hdw.reshape(-1, 3))
                        .reshape(hdw.shape[:2])
                        .astype(np.float32)
                    )
                    # normalize and invert the colors because pca produces mainly negative images (?)
                    # -> work because we use the abs(pca)
                    hdw = (hdw.max() - hdw) / (hdw.max() - hdw.min())
                    # -> classic min-max scaling
                    # hdw = (hdw - hdw.min()) / (hdw.max() - hdw.min())

                hdw = self.transform(hdw) if self.transform else hdw
                cad = self.getLabel(
                    self.transform(cad) if self.transform else self.getLabel(cad)
                )

                if self.limit_memory:
                    # write the database
                    self.cache_fileobj.create_dataset(
                        f"hdw_{i}", data=hdw, compression="gzip"
                    )
                    self.cache_fileobj.create_dataset(
                        f"cad_{i}", data=cad, compression="gzip"
                    )
                else:
                    self.data_hdw.append(hdw)
                    self.data_cad.append(cad)

        # compute memory usage and leave
        if not self.limit_memory:
            print("mem usage: %.3f GB" % self.get_mem())

        cad = None
        hdw = None
        if self.Patch:
            print("** Removing tmp patch cache files.")
            self.f_cad.close()
            self.f_hdw.close()
            os.remove(".hdw.tmp")
            os.remove(".cad.tmp")

    def get_mem(self):
        return (
            (
                sum([di.nbytes for di in self.data_hdw])
                + sum([dl.nbytes for dl in self.data_cad])
            )
            / 1024**3
            if not self.limit_memory
            else 0
        )

    def getLabel(self, target):
        """From the target computes the label for classification"""
        # < eps -> label is 1 (class positive) for black pixels (0 intensity) and 0 (class negative) for white pixels (255 intensity)
        # >= eps -> label is 0 for black pixels and 1 for white pixels

        # should be 255 or max(target) ?
        # if isinstance(target, torch.Tensor):
        # kind of an issue here, without the round the condition is False
        # making the code understand it is a numpy array. We don't want to import torch here.

        if self.eps <= 0:
            return target

        if isinstance(target, np.ndarray):
            label = (
                (1 * (target < self.eps)).astype(np.uint8)
                if (np.round(target.max(), 3) <= 1.0)
                else (1 * (target < int(255 * self.eps))).astype(np.uint8)
            )
        else:  # presumably some kind of normalized tensor
            label = 1.0 * (target < self.eps)

        return label

    def __len__(self):
        """Return the len of the data generator."""
        return self.num_samples

    def __getitem_from_cache__(self, index, slices=None):
        with h5py.File(self.cache_file, "r") as self.cache_fileobj:
            hdw = (
                np.array(self.cache_fileobj[f"hdw_{index}"][slices])
                if slices
                else np.array(self.cache_fileobj[f"hdw_{index}"])
            )
            cad = (
                np.array(self.cache_fileobj[f"cad_{index}"][slices])
                if slices
                else np.array(self.cache_fileobj[f"cad_{index}"])
            )
        return hdw, cad

    def __getitem__(self, idx):
        """Access the next item in the generator with index idx.
        if self.use_patch, idx corresponds to any patch of any image.
        Otherwise, idx is the index of any image.

        Args:
            idx (int): index of elements in the generator

        Returns:
            ndarray, ndarray: inputs and labels
        """
        # directly load from the pre-compiled database the input and label
        # for either patch or full image
        if self.limit_memory:
            return self.__getitem_from_cache__(idx)
        # Or access to the transformed data by memory access
        else:
            return self.data_hdw[idx], self.data_cad[idx]


# from glob import glob
# patch = Patch(size=256, step=None)

# hdw_files = glob("/home/DataSets/EDF_DATA/edf_plans2D_old/*")[:1]
# cad_files = glob("/home/DataSets/EDF_DATA/edf_clean_base/*.png")[:2]
