using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OutlineHover : MonoBehaviour
{
    private bool HasPointer;
    private Outline outline;

    public void PointerEnter()
    {
        HasPointer = true;
    }

    public void PointerExit()
    {
        HasPointer = false;
    }

    // Start is called before the first frame update
    void Start()
    {
        outline = GetComponentInChildren<Outline>();

        outline.OutlineWidth = 8;
    }

    // Update is called once per frame
    void Update()
    {
        if (HasPointer)
        {
            outline.OutlineColor = Color.blue;
        }
        else
        {
            outline.OutlineColor = new Color(0, 0, 0, 0);
        }
    }
}
