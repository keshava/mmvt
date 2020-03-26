import importlib

def run(mmvt):
    mu = mmvt.utils
    mu.add_mmvt_code_root_to_path()
    from src.preproc import anatomy as anat
    importlib.reload(anat)
    anat.create_surface_volume_mask(mu.get_user(), 'cerebellum')
