IMG_MK_INC = img/convol/convol.mk
IMG_MK_INC += img/dspk/dspk.mk
IMG_MK_INC += img/util/util.mk

include $(IMG_MK_INC)

KSO_OBJS += $(IMG_OBJS)
KSO_ARTS += $(IMG_ARTS)


