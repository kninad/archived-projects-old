using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Photon.Pun;

public class DisplayBuildMode : MonoBehaviour
{
    public Text buildPrompt;
    public Text modeText;
    public static string modeTexttxt;


    // Start is called before the first frame update
    void Start()
    {
        buildPrompt.text = "Prompt: " + PhotonNetwork.CurrentRoom.Name;
        //modeTexttxt = "Current Mode: Create";
        //modeText.text = modeTexttxt;
    }

    // Update is called once per frame
    void Update()
    {
        modeText.text = modeTexttxt;
    }
}
