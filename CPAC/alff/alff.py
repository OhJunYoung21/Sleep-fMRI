import os
from CPAC.pipeline import nipype_pipeline_engine as pe
from CPAC.pipeline.nodeblock import nodeblock
from nipype.interfaces.afni import preprocess
import nipype.interfaces.utility as util
from CPAC.alff.utils import get_opt_string
from CPAC.utils.utils import check_prov_for_regtool
from CPAC.registration.registration import apply_transform


def create_alff(wf_name='alff_workflow'):
    """
    Calculate Amplitude of low frequency oscillations (ALFF) and fractional ALFF maps

    Parameters
    ----------
    wf_name : string
        Workflow name

    Returns
    -------
    alff_workflow : workflow object
        ALFF workflow

    Notes
    -----
    `Source <https://github.com/FCP-INDI/C-PAC/blob/master/CPAC/alff/alff.py>`_

    Workflow Inputs::

        hp_input.hp : list of float
            high pass frequencies

        lp_input.lp : list of float
            low pass frequencies

        inputspec.rest_res : string
            Path to existing Nifti file. Nuisance signal regressed functional image.

        inputspec.rest_mask : string
            Path to existing Nifti file. A mask volume(derived by dilating the motion corrected functional volume) in native space


    Workflow Outputs::

        outputspec.alff_img : string
            Path to Nifti file. Image containing the sum of the amplitudes in the low frequency band

        outputspec.falff_img : string
            Path to Nifti file. Image containing the sum of the amplitudes in the low frequency band divided by the amplitude of the total frequency

        outputspec.alff_Z_img : string
            Path to Nifti file. Image containing Normalized ALFF Z scores across full brain in native space

        outputspec.falff_Z_img : string
            Path to Nifti file. Image containing Normalized fALFF Z scores across full brain in native space


    Order of Commands:

    - Filter the input file rest file( slice-time, motion corrected and nuisance regressed) ::
        3dBandpass -prefix residual_filtered.nii.gz
                    0.009 0.08 residual.nii.gz

    - Calculate ALFF by taking the standard deviation of the filtered file ::
        3dTstat -stdev
                -mask rest_mask.nii.gz
                -prefix residual_filtered_3dT.nii.gz
                residual_filtered.nii.gz

    - Calculate the standard deviation of the unfiltered file ::
        3dTstat -stdev
                -mask rest_mask.nii.gz
                -prefix residual_3dT.nii.gz
                residual.nii.gz

    - Calculate fALFF ::
        3dcalc -a rest_mask.nii.gz
               -b residual_filtered_3dT.nii.gz
               -c residual_3dT.nii.gz
               -expr '(1.0*bool(a))*((1.0*b)/(1.0*c))' -float

    - Normalize ALFF/fALFF to Z-score across full brain ::

        fslstats
        ALFF.nii.gz
        -k rest_mask.nii.gz
        -m > mean_ALFF.txt ; mean=$( cat mean_ALFF.txt )

        fslstats
        ALFF.nii.gz
        -k rest_mask.nii.gz
        -s > std_ALFF.txt ; std=$( cat std_ALFF.txt )

        fslmaths
        ALFF.nii.gz
        -sub ${mean}
        -div ${std}
        -mas rest_mask.nii.gz ALFF_Z.nii.gz

        fslstats
        fALFF.nii.gz
        -k rest_mask.nii.gz
        -m > mean_fALFF.txt ; mean=$( cat mean_fALFF.txt )

        fslstats
        fALFF.nii.gz
        -k rest_mask.nii.gz
        -s > std_fALFF.txt
        std=$( cat std_fALFF.txt )

        fslmaths
        fALFF.nii.gz
        -sub ${mean}
        -div ${std}
        -mas rest_mask.nii.gz
        fALFF_Z.nii.gz

    .. exec::
        from CPAC.alff import create_alff
        wf = create_alff()
        wf.write_graph(
            graph2use='orig',
            dotfilename='./images/generated/alff.dot'
        )

    High Level Workflow Graph:

    .. image:: ../../images/generated/alff.png
        :width: 500

    Detailed Workflow Graph:

    .. image:: ../../images/generated/alff_detailed.png
        :width: 500


    References
    ----------

    .. [1] Zou, Q.-H., Zhu, C.-Z., Yang, Y., Zuo, X.-N., Long, X.-Y., Cao, Q.-J., Wang, Y.-F., et al. (2008). An improved approach to detection of amplitude of low-frequency fluctuation (ALFF) for resting-state fMRI: fractional ALFF. Journal of neuroscience methods, 172(1), 137-41. doi:10.10

    Examples
    --------

    >>> alff_w = create_alff()
    >>> alff_w.inputs.hp_input.hp = [0.01]
    >>> alff_w.inputs.lp_input.lp = [0.1]
    >>> alff_w.get_node('hp_input').iterables = ('hp', [0.01])
    >>> alff_w.get_node('lp_input').iterables = ('lp', [0.1])
    >>> alff_w.inputs.inputspec.rest_res = '/home/data/subject/func/rest_bandpassed.nii.gz'
    >>> alff_w.inputs.inputspec.rest_mask= '/home/data/subject/func/rest_mask.nii.gz'
    >>> alff_w.run() # doctest: +SKIP
    """

    wf = pe.Workflow(name=wf_name)
    input_node = pe.Node(util.IdentityInterface(fields=['rest_res',
                                                        'rest_mask']),
                         name='inputspec')

    input_node_hp = pe.Node(util.IdentityInterface(fields=['hp']),
                            name='hp_input')

    input_node_lp = pe.Node(util.IdentityInterface(fields=['lp']),
                            name='lp_input')

    output_node = pe.Node(util.IdentityInterface(fields=['alff_img',
                                                         'falff_img']),
                          name='outputspec')

    # filtering
    bandpass = pe.Node(interface=preprocess.Bandpass(),
                       name='bandpass_filtering')
    bandpass.inputs.outputtype = 'NIFTI_GZ'
    bandpass.inputs.out_file = os.path.join(os.path.curdir,
                                            'residual_filtered.nii.gz')

    wf.connect(input_node_hp, 'hp', bandpass, 'highpass')
    wf.connect(input_node_lp, 'lp', bandpass, 'lowpass')
    wf.connect(input_node, 'rest_res', bandpass, 'in_file')

    get_option_string = pe.Node(util.Function(input_names=['mask'],
                                              output_names=['option_string'],
                                              function=get_opt_string),
                                name='get_option_string')

    wf.connect(input_node, 'rest_mask', get_option_string, 'mask')

    # standard deviation over frequency
    try:
        from nipype.interfaces.afni import utils as afni_utils
        stddev_filtered = pe.Node(interface=afni_utils.TStat(),
                                  name='stddev_filtered')
    except ImportError:
        stddev_filtered = pe.Node(interface=preprocess.TStat(),
                                  name='stddev_filtered')

    stddev_filtered.inputs.outputtype = 'NIFTI_GZ'
    stddev_filtered.inputs.out_file = os.path.join(os.path.curdir,
                                                   'alff.nii.gz')

    wf.connect(bandpass, 'out_file', stddev_filtered, 'in_file')
    wf.connect(get_option_string, 'option_string', stddev_filtered, 'options')

    wf.connect(stddev_filtered, 'out_file', output_node, 'alff_img')

    # standard deviation of the unfiltered nuisance corrected image
    try:
        stddev_unfiltered = pe.Node(interface=afni_utils.TStat(),
                                    name='stddev_unfiltered')
    except UnboundLocalError:
        stddev_unfiltered = pe.Node(interface=preprocess.TStat(),
                                    name='stddev_unfiltered')

    stddev_unfiltered.inputs.outputtype = 'NIFTI_GZ'
    stddev_unfiltered.inputs.out_file = os.path.join(os.path.curdir,
                                                     'residual_3dT.nii.gz')

    wf.connect(input_node, 'rest_res', stddev_unfiltered, 'in_file')
    wf.connect(get_option_string, 'option_string', stddev_unfiltered,
               'options')

    # falff calculations
    try:
        falff = pe.Node(interface=afni_utils.Calc(), name='falff')
    except UnboundLocalError:
        falff = pe.Node(interface=preprocess.Calc(), name='falff')

    falff.inputs.args = '-float'
    falff.inputs.expr = '(1.0*bool(a))*((1.0*b)/(1.0*c))'
    falff.inputs.outputtype = 'NIFTI_GZ'
    falff.inputs.out_file = os.path.join(os.path.curdir, 'falff.nii.gz')

    wf.connect(input_node, 'rest_mask', falff, 'in_file_a')
    wf.connect(stddev_filtered, 'out_file', falff, 'in_file_b')
    wf.connect(stddev_unfiltered, 'out_file', falff, 'in_file_c')

    wf.connect(falff, 'out_file', output_node, 'falff_img')

    return wf


@nodeblock(
    name="alff_falff",
    config=["amplitude_low_frequency_fluctuation"],
    switch=["run"],
    inputs=[
        (
                ["desc-denoisedNofilt_bold", "desc-preproc_bold"],
                "space-bold_desc-brain_mask",
        )
    ],
    outputs=["alff", "falff"],
)
def alff_falff(wf, cfg, strat_pool, pipe_num, opt=None):
    alff = create_alff(f'alff_falff_{pipe_num}')

    alff.inputs.hp_input.hp = \
        cfg.amplitude_low_frequency_fluctuation['highpass_cutoff']
    alff.inputs.lp_input.lp = \
        cfg.amplitude_low_frequency_fluctuation['lowpass_cutoff']
    alff.get_node('hp_input').iterables = ('hp', alff.inputs.hp_input.hp)
    alff.get_node('lp_input').iterables = ('lp', alff.inputs.lp_input.lp)

    node, out = strat_pool.get_data(["desc-denoisedNofilt_bold",
                                     "desc-preproc_bold"])
    wf.connect(node, out, alff, 'inputspec.rest_res')

    node, out = strat_pool.get_data('space-bold_desc-brain_mask')
    wf.connect(node, out, alff, 'inputspec.rest_mask')

    outputs = {
        'alff': (alff, 'outputspec.alff_img'),
        'falff': (alff, 'outputspec.falff_img')
    }

    return (wf, outputs)


@nodeblock(
    name="alff_falff_space_template",
    config=["amplitude_low_frequency_fluctuation"],
    switch=["run"],
    inputs=[
        (
                [
                    "space-template_res-derivative_desc-denoisedNofilt_bold",
                    "space-template_res-derivative_desc-preproc_bold",
                    "space-template_desc-preproc_bold"
                ],
                [
                    "space-template_res-derivative_desc-bold_mask",
                    "space-template_desc-bold_mask"
                ],
                "desc-denoisedNofilt_bold",
                "from-bold_to-template_mode-image_xfm",
                "T1w-brain-template-deriv",
        )
    ],
    outputs=["space-template_alff", "space-template_falff",
             "space-template_res-derivative_desc-denoisedNofilt_bold"],
)
def alff_falff_space_template(wf, cfg, strat_pool, pipe_num, opt=None):
    outputs = {}
    if strat_pool.check_rpool("desc-denoisedNofilt_bold"):
        xfm_prov = strat_pool.get_cpac_provenance(
            'from-bold_to-template_mode-image_xfm')
        reg_tool = check_prov_for_regtool(xfm_prov)

        num_cpus = cfg.pipeline_setup['system_config'][
            'max_cores_per_participant']

        num_ants_cores = cfg.pipeline_setup['system_config']['num_ants_threads']

        apply_xfm = apply_transform(f'warp_denoisedNofilt_to_T1template_{pipe_num}', reg_tool,
                                    time_series=True, num_cpus=num_cpus,
                                    num_ants_cores=num_ants_cores)

        if reg_tool == 'ants':
            apply_xfm.inputs.inputspec.interpolation = cfg.registration_workflows[
                'functional_registration']['func_registration_to_template'][
                'ANTs_pipelines']['interpolation']
        elif reg_tool == 'fsl':
            apply_xfm.inputs.inputspec.interpolation = cfg.registration_workflows[
                'functional_registration']['func_registration_to_template'][
                'FNIRT_pipelines']['interpolation']

        node, out = strat_pool.get_data("desc-denoisedNofilt_bold")
        wf.connect(node, out, apply_xfm, 'inputspec.input_image')

        node, out = strat_pool.get_data("T1w-brain-template-deriv")
        wf.connect(node, out, apply_xfm, 'inputspec.reference')

        node, out = strat_pool.get_data("from-bold_to-template_mode-image_xfm")
        wf.connect(node, out, apply_xfm, 'inputspec.transform')

        outputs = {
            f'space-template_res-derivative_desc-denoisedNofilt_bold': (apply_xfm, 'outputspec.output_image')
        }
    alff = create_alff(f'alff_falff_{pipe_num}')

    alff.inputs.hp_input.hp = \
        cfg.amplitude_low_frequency_fluctuation['highpass_cutoff']
    alff.inputs.lp_input.lp = \
        cfg.amplitude_low_frequency_fluctuation['lowpass_cutoff']
    alff.get_node('hp_input').iterables = ('hp', alff.inputs.hp_input.hp)
    alff.get_node('lp_input').iterables = ('lp', alff.inputs.lp_input.lp)
    node, out = strat_pool.get_data(["space-template_res-derivative_desc-denoisedNofilt_bold",
                                     "space-template_res-derivative_desc-preproc_bold",
                                     "space-template_desc-preproc_bold"])
    wf.connect(node, out, alff, 'inputspec.rest_res')
    node, out = strat_pool.get_data(["space-template_res-derivative_desc-bold_mask",
                                     "space-template_desc-bold_mask"])
    wf.connect(node, out, alff, 'inputspec.rest_mask')

    outputs.update({
        'space-template_alff': (alff, 'outputspec.alff_img'),
        'space-template_falff': (alff, 'outputspec.falff_img')
    })

    return (wf, outputs)

### shit
