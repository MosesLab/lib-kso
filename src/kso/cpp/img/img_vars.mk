IMG_MK_VARS += img/dspk/dspk_vars.mk
IMG_MK_VARS += img/convol/convol_vars.mk

KSO_OBJS += $(IMG_OBJS)
KSO_ARTS += $(IMG_ARTS)

include $(IMG_MK_VARS)
