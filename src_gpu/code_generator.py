#!/usr/bin/python
"""
unroll certain for loops and generate C++ code
"""

def ao_ordering(l):
    """list of Cartesian basis function with angular momentum l"""
    if l == 0:
        return [(0,0,0)]
    elif l == 1:
        return [(1,0,0), (0,1,0), (0,0,1)]
    elif l == 2:
        # ordering of d-functions in TeraChem: dxy,dxz,dyz,dxx,dyy,dzz
        return [(1,1,0), (1,0,1), (0,1,1), (2,0,0), (0,2,0), (0,0,2)]
    else:
        # The integral routines work for any angular momentum, but what ordering should
        # be used for f,g,etc. functions?
        raise NotImplemented("Angular momenta higher than s,p and d functions are not implemented!")


def enumerate_angular_momenta(lmax=2):
    """
    generate code for function `polarization_prim_pairs_kernel`
    """
    l2s = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}

    code = "/**** BEGIN of automatically generated code (with code_generator.py) *****/\n"
    for lA in range(0, lmax+1):
        if lA == 0:
            code += f"  if (lA == {lA}) {{\n"
        else:
            code += f"  }} else if (lA == {lA}) {{\n"
        for lB in range(0, lmax+1):
            if lB == 0:
                code += f"    if (lB == {lB}) {{"
            else:
                code += f"    }} else if (lB == {lB}) {{"
            code += f"""
      // {l2s[lA]}{l2s[lB]} integrals
      PolarizationIntegral<real, {lA}, {lB}, k, mx, my, mz, q> integrals
        (primA->x, primA->y, primA->z, primA->exp,
         primB->x, primB->y, primB->z, primB->exp,
         alpha);
            """
            nB = len(ao_ordering(lB))
            for i,(nxi,nyi,nzi) in enumerate(ao_ordering(lA)):
                for j,(nxj,nyj,nzj) in enumerate(ao_ordering(lB)):
                    code += f"""
      buffer[ij+{i:2}*{nB:2}+{j:2}] = cc * integrals.template compute_pair<{nxi},{nyi},{nzi}, {nxj},{nyj},{nzj}>();"""
            code += f"\n"
        code += f"    }}\n"
    code += f"  }}\n"

    code += "/**** END of automatically generated code *****/\n"
            
    return code

def partition3(l):
    """
    enumerate all partitions of the integer l into 3 integers nx, ny, nz
    such that nx+ny+nz = l
    """
    for nx in range(0, l+1):
        for ny in range(0, l-nx+1):
            nz = l-nx-ny
            yield (nx,ny,nz)

def template_declarations():
    """
    Because the declaration and definition of the templates are in different files
    we have to list the template specialization for which code should be compiled
    at the end of the *.cu file
    """
    code = "/**** BEGIN of automatically generated code (with code_generator.py) *****/\n"
    # If you need integrals for other values of k,mx,my,mz or q just add a declaration.

    def declaration(k, mx,my,mz):
        return f"""
template void polarization_prim_pairs<double, {k},   {mx}, {my}, {mz},   {q}>
        (const PrimitivePair<double> *pairs, int npair, double *buffer, double alpha);
template void polarization_prim_pairs<float,  {k},   {mx}, {my}, {mz},   {q}>
        (const PrimitivePair<float> *pairs, int npair, float *buffer, float alpha);
"""

    # cutoff power
    q = 2
    # enumerate all polarization operators 
    #   Op(x,y,z) = x^mx y^my z^mz / |r|^k   
    # for which we need integrals

    # 1/r^3
    code += "\n// Op(r) = 1/|r|^3\n"
    code += declaration(3, 0,0,0)   
    # 1/r^4
    code += "\n// Op(r) = 1/|r|^4\n"
    code += declaration(4, 0,0,0)   
    # r(i)/r^3
    code += "\n// Op(r) = r(i)/|r|^3\n"
    for mx,my,mz in partition3(1):
        code += declaration(3, mx,my,mz)
    # r(i)r(j)/r^6   
    code += "\n// Op(r) = r(i)r(j)/|r|^6\n"
    for mx,my,mz in partition3(2):
        code += declaration(6, mx,my,mz)

    code += "/**** END of automatically generated code *****/\n"

    return code

def runtime_template_selection():
    """
    Suppose a template is instantiated for all possible combinations of integer parameters
    for the function polarization_prim_pairs<..., k,mx,my,mz, q>. To select
    the appropriate template at runtime, code wit nested switch clauses is generated
    automatically.

    The generated code has to be inserted in `polarization_prim_pairs_wrapper()`.
    """
    # highest value that any of the integers (k-3), mx, my, mz and (q-2) 
    # can take, 4 should be more than enough
    N = 4
    code = f"""
  /**** BEGIN of automatically generated code (with code_generator.py) *****/
  // highest value that any of the integers (k-3), mx, my, mz and (q-2) can take
  const int N = {N};
  // The index of the template instance for <...,k,mx,my,mz,q>
  int template_instance = (k-3)*N*N*N*N + mx*N*N*N + my*N*N + mz*N + q-2;
  switch (template_instance) {{
"""
    def cases(k, l):
        code = ""
        for (mx,my,mz) in partition3(l):
            # index of template
            template_instance = (k-3)*N*N*N*N + mx*N*N*N + my*N*N + mz*N + q-2
            code += f"""  case {template_instance}:
    polarization_prim_pairs<real, {k}, {mx},{my},{mz}, {q}>(pairs, npair, buffer, alpha); break;
"""
        return code

    # cutoff power
    q = 2

    # 1/r^3
    code += "\n  // Op(r) = 1/|r|^3\n"
    code += cases(3, 0)
    # 1/r^4
    code += "\n  // Op(r) = 1/|r|^4\n"
    code += cases(4, 0)
    # r(i)/r^3
    code += "\n  // Op(r) = r(i)/|r|^3\n"
    code += cases(3, 1)
    # r(i)r(j)/r^6   
    code += "\n  // Op(r) = r(i)r(j)/|r|^6\n"
    code += cases(6, 2)

    code += """  default:
    throw std::runtime_error(\"No template instance found for this combination of k, mx,my,mz, q ! Add it and recompile. \");
  }
  /**** END of automatically generated code *****/
"""
    return code

if __name__ == "__main__":
    #print(enumerate_angular_momenta(lmax=2))
    #print(template_declarations())
    print(runtime_template_selection())
