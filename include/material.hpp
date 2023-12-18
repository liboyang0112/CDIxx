class BaseMaterial{
  public:
    BaseMaterial(){};
    virtual double getRefractiveIndex(double lambda) = 0;
    virtual double getExtinctionLength(double lambda) = 0;
};

class ToyMaterial : BaseMaterial{
  public:
    ToyMaterial() : BaseMaterial(){};
    double getRefractiveIndex(double lambda);
    double getExtinctionLength(double lambda);
};

class TabledMaterial : BaseMaterial{
  double *refractiveIdx;
  double *extinctionLen;
  public:
    TabledMaterial() : BaseMaterial(), refractiveIdx(0), extinctionLen(0) {};
    TabledMaterial(double* a, double* b) : BaseMaterial(), refractiveIdx(a), extinctionLen(b) {};
    void setData(double* a, double* b){refractiveIdx = a; extinctionLen = b;}
    double getRefractiveIndex(double lambda);
    double getExtinctionLength(double lambda);
};
